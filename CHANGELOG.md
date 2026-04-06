# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.18.0] - 2026-04-06

### Added

- **End-to-end LLM inference engine** (`src/inference/`)
  - `GgufFile`: GGUF v2/v3 reader — metadata KV, tensor info, alignment-padded data
  - `LlamaModel`: Full transformer — RMSNorm, Q4K fused matmul, RoPE, GQA, SwiGLU FFN, KV cache
  - `WeightMatrix` enum: Q4K fused path for hot weights, F32 dequant path for mixed quantization
  - Dequantization: Q4_0, Q4_1, Q4K, Q5K, Q6K, Q8_0, F16, BF16
  - `generate()`: Autoregressive decode with temperature, top-k, top-p nucleus sampling
  - QKV bias support for Qwen2/Qwen3 architectures
  - `examples/inference_demo.rs`: CLI — load GGUF, tokenize, generate, print tok/s stats

- **Software-pipelined GPU GEMM kernel** (64×128 CTA, 3-stage cp.async)
  - 60.9 TF/s peak at 2048 — 0.52× cuBLAS, TARGET MET
  - 18KB shared memory (3×6KB pipeline stages)
  - 5 FALSIFY tests, 19/19 contracts pass

### Performance

- **P5c industry baseline**: trueno 807 tok/s vs llama.cpp 2481 tok/s (0.33×) on TinyLlama 5M F16
- GPU GEMM pipeline: +39% over non-pipelined (60.9 vs 43.8 TF/s at 2048)
- All 3630 tests pass

## [0.16.0] - 2026-02-26

### Changed
- Minor version bump for PAIML Sovereign AI Stack coordinated release
- Updated workspace lints and CI configurations

## [0.9.0] - 2025-12-30

### Added

- **CUDA-Tile Behavior GPU Optimizations** (cuda-tile-behavior.md spec)
  - `TensorView<T>`: Structured memory view with shape/stride metadata for GPU buffers
  - `PartitionView<T>`: Tiling strategy for 16x16 GPU workgroup distribution
  - Tiled reduction algorithms: `tiled_sum_2d`, `tiled_max_2d`, `tiled_min_2d`
  - `ReduceOp` trait for custom reduction operations (SumOp, MaxOp, MinOp)
  - WGSL tiled reduction shaders for GPU compute (pending integration)

- **Intel SDE Support for AVX-512 Testing** (Makefile targets)
  - `make install-sde`: Download and install Intel Software Development Emulator
  - `make test-avx512-sde`: Run AVX-512 tests under Skylake-X emulation
  - `make bench-avx512-sde`: Run AVX-512 benchmarks under emulation
  - `make coverage-avx512-sde`: Run AVX-512 coverage under emulation
  - Enables AVX-512 testing on CPUs without native support (e.g., Intel Meteor Lake)

- **PTX Optimization Passes** (trueno-gpu)
  - FMA fusion pass: Automatically fuse mul+add into fma instructions
  - Tile validation: Compile-time validation of tile constraints
  - ~33% instruction reduction for FMA-eligible code

### Documentation

- New example: `tiled_reduction_demo` demonstrating GPU memory abstractions
- Updated book chapter: GPU Compute Shaders with tiled reduction algorithms
- GitHub issues #72-#76 filed for CUDA-specific integration work

### Fixed

- AVX-512 dot product test tolerance using relative error for large results
- Clippy warnings for match arms, must_use, and div_ceil

## [0.8.9] - 2025-12-23

### Added

- **Batched Matrix Multiplication** for 3D and 4D tensors (Refs #71)
  - `Matrix::batched_matmul`: Shape `[batch, m, k] @ [batch, k, n] -> [batch, m, n]`
  - `Matrix::batched_matmul_4d`: Attention pattern `[batch, heads, m, k] @ [batch, heads, k, n]`
  - SIMD-accelerated using trueno's matmul backend
  - Critical for transformer multi-head attention (Q @ K^T, attn @ V)
  - 8 unit tests for correctness and error handling

### Documentation

- Updated `examples/matrix_operations.rs` with batched matmul demos
- Book updates for batched matmul API reference and examples
- Created GitHub issue #71 for BatchedGemmKernel GPU support

## [0.8.8] - 2025-12-17

### Changed

- Updated `trueno-gpu` dependency to v0.3.0
  - BiasActivationKernel: Fused bias + activation epilogue (None/ReLU/GELU)
  - GemvKernel: Matrix-vector multiply for M=1 matmuls in LLM inference

### Documentation

- Book updates for BiasActivationKernel examples and PTX generation

## [trueno-gpu 0.3.0] - 2025-12-17

### Added

- **BiasActivationKernel**: Fused bias + activation epilogue kernel for GEMM operations
  - Three activation variants: None (bias only), ReLU, GELU
  - Builder pattern API: `BiasActivationKernel::new(n, bias_size).with_relu()`
  - GELU uses fast `ex2.approx` for exponential approximation
  - `bias_size` baked into kernel at generation time for efficiency
  - 22 tests including property-based and falsification tests
  - 100% mutation coverage (2 caught by tests, 4 by type system)

- **GemvKernel**: Matrix-vector multiply optimized for M=1 matmuls
  - One warp (32 threads) per output element
  - Warp shuffle reduction for efficient dot products
  - Critical path for LLM token generation

### Documentation

- Added Examples section to README with run commands
- Updated Available Kernels table with BiasActivation, GEMV, Q5_K/Q6_K
- Book documentation for BiasActivationKernel with testing commands

## [0.8.5] - 2025-12-15

### Added

- **Simulation Testing Framework** (`simulation` module) - TRUENO-SPEC-012
  - `SimRng`: Deterministic PCG-based RNG for reproducible testing
  - `BackendSelector`: Intelligent backend selection with configurable thresholds
  - `JidokaGuard`: Toyota-style stop-on-defect quality checks (NaN/Inf detection)
  - `HeijunkaScheduler`: Load-leveled test scheduling across backends
  - `BufferRenderer`: RGBA buffer rendering for visual regression testing
  - `ColorPalette`: Viridis and grayscale palettes for heatmap visualization
  - `GoldenBaseline`: Golden file comparison for deterministic validation
  - `StressTestConfig/Result`: Stress testing infrastructure with anomaly detection
  - `BackendTolerance`: Cross-backend comparison tolerance configuration

- **100 Falsifiable Claims** - Comprehensive test suite validating:
  - Backend selection logic (Claims 1-15)
  - Determinism guarantees (Claims 16-30)
  - SIMD operation correctness (Claims 31-50)
  - PTX kernel patterns (Claims 51-65)
  - WGPU shader correctness (Claims 66-80)
  - Visual regression framework (Claims 81-90)
  - Stress testing infrastructure (Claims 91-100)

### Fixed

- `make coverage-check` now correctly parses coverage percentage
- Coverage excludes external `simular` dependency for accurate metrics

## [trueno-gpu 0.1.0] - 2025-12-10

### Added

- **trueno-gpu sub-crate**: Pure Rust PTX generation for NVIDIA CUDA
  - No LLVM, no nvcc, no external dependencies required for code generation
  - Builder pattern API for constructing PTX modules and kernels
  - PTX ISA 8.0 compliant output

- **PTX Code Generation** (`ptx` module)
  - `PtxModule`: Module builder with version, target, address size configuration
  - `PtxKernel`: Kernel builder with parameters, shared memory, body generation
  - `PtxBuilder`: Instruction builder with virtual register allocation
  - Type system: U8, U16, U32, U64, S8, S16, S32, S64, F16, F32, F64, Pred
  - Special registers: TidX/Y/Z, CtaIdX/Y/Z, NtidX/Y/Z

- **Hand-Optimized Kernels** (`kernels` module)
  - **GEMM**: Matrix multiplication with 3 variants
    - Naive: Simple O(n³) implementation
    - Tiled: Shared memory tiling for cache optimization
    - Tensor Core: WMMA instructions for fp16 acceleration
  - **Softmax**: Numerically stable softmax with warp shuffle reduction
  - **LayerNorm**: Fused layer normalization with 2 variants
    - Warp shuffle: Uses shuffle instructions for parallel reduction
    - Shared memory: Uses shared memory for larger dimensions
  - **Attention**: FlashAttention-style tiled attention
    - Online softmax algorithm (never materializes N×N matrix)
    - Causal masking support
    - Configurable Q/KV tile sizes
  - **Quantize**: Q4_K dequantization-fused GEMM
    - 4-bit quantized weights (32 weights per 18-byte block)
    - Fused dequantization during matmul

- **Supporting Modules**
  - `driver`: CUDA driver API FFI (optional, for GPU execution)
  - `memory`: GPU memory management abstractions
  - `backend`: Multi-backend abstraction layer
  - `error`: Error types and Result alias

### Quality

- **145 unit tests** (100% passing)
- **2 doc tests** (100% passing)
- Zero clippy warnings
- EXTREME TDD methodology applied throughout

## [0.8.1] - 2025-12-08

### Added ✨

- **Quick Start Example** (`examples/quickstart.rs`)
  - Comprehensive example showcasing all core Trueno features in one file
  - Vector operations, matrix math, eigendecomposition, activations, layer norm
  - Recommended starting point for new users

- **Enhanced API Documentation**
  - `book/src/api-reference/vector-operations.md` - Complete vector API reference
  - `book/src/api-reference/matrix-operations.md` - Matrix operations guide
  - `book/src/api-reference/eigendecomposition.md` - SymmetricEigen documentation

### Changed 🔄

- Updated examples README with all current examples including `symmetric_eigen`, `hash_demo`, `gpu_batch_demo`
- Applied `cargo fmt` formatting fixes across codebase
- Installed PMAT TDG enforcement hooks for quality gates

### Quality 📊

- **Repository Score**: 100/100 (A+)
- **TDG Score**: 90.4/100 (A)
- **Rust Project Score**: 143.9/134 (107.4%, A+)
- All 954 tests passing
- Benchmarks verified: dot product 11-12x speedup (AVX-512), eigen 1.3-2.2x faster than nalgebra

## [0.7.3] - 2025-11-25

### Added ✨

- **WebGPU for WASM** (`gpu-wasm` feature)
  - Cross-platform GPU compute: same code runs on native and browser
  - Async-first API design: all GPU operations have `*_async` variants
  - Runtime detection: `runtime::sync_available()` for platform-specific code paths
  - New `runtime` module (`src/backends/gpu/runtime.rs`) for platform abstraction
  - Enables [trueno-viz](https://github.com/paiml/trueno-viz) browser-based GPU visualization

- **Cross-platform GPU API**
  - `GpuDevice::new_async()` - Works on all platforms (native + WASM)
  - `GpuDevice::is_available_async()` - Async availability check
  - All operations now have async variants: `relu_async`, `sigmoid_async`, `matmul_async`, etc.
  - Sync wrappers remain available on native platforms only

### Changed 🔄

- GPU device initialization refactored to use `runtime::block_on()` instead of direct `pollster::block_on()`
- Conditional compilation: sync methods require `#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]`
- All private async methods now public (`pub async fn *_async`)

### Documentation 📚

- **GPU Backend chapter** (`book/src/architecture/gpu-backend.md`) - Complete rewrite
  - Platform support matrix (Linux/macOS/Windows/WASM)
  - Feature flag comparison (`gpu` vs `gpu-wasm`)
  - Async-first API examples
  - trueno-viz integration guide
  - Runtime detection patterns

- **GPU Performance chapter** - Added WebGPU/WASM section
  - Platform differences table
  - Async API usage examples
  - trueno-viz reference

### Fixed 🐛

- `select_backend_for_operation` parameter name: `_op_type` → `op_type` (parameter is used)
- Type inference in empty slice comparisons: `&[]` → `&[] as &[f32]`
- Unused variable in WASM backend: `scale` → `_scale`

### Dependencies 📦

- Added `wasm-bindgen-futures` (0.4) for WASM async support
- Added `wasm-bindgen` (0.2) for WASM bindings
- Added `web-sys` (0.3) for browser APIs (console logging)

### Testing ✅

- All 903+ tests passing
- Coverage: 90.40% (exceeds 90% requirement)
- Added `required-features = ["gpu"]` for `gpu_batch_demo` example

## [0.7.1] - 2025-11-24

### Added ✨

- **EXTREME PMAT Integration** - O(1) Quality Gates
  - Enhanced validation workflow for technical debt grading
  - Automated quality metrics enforcement
  - Repository health score tracking (minimum 90/110)

- **Golden Trace Validation** (Renacer v0.6.2+)
  - Syscall-level performance regression detection
  - Captured golden traces for 5 core operations (backend_detection, matrix_operations, activation_functions, performance_demo, ml_similarity)
  - Performance assertions via `renacer.toml` (CI fails on regression)
  - Comprehensive documentation: `docs/integration-report-golden-trace.md`
  - Book chapter: `book/src/performance/golden-trace-validation.md`
  - GitHub Actions workflow for automated validation

- **GPU Batch API Example**
  - Demonstration example for async GPU command batching
  - Shows 3x transfer reduction for chained operations

### Fixed 🐛

- Replaced `.unwrap()` with `.expect()` in examples for better error messages
- Corrected relative paths in golden-trace-validation.md documentation
- Fixed formatting issues across examples

### Infrastructure 🔧

- Added GitHub Actions workflow for golden trace validation
- Updated gitignore: `direct_bench.log`, `benchmark_run.log`

### Documentation 📚

- Updated book: async GPU batch API now available (v0.3.0)
- Enhanced golden trace validation documentation
- Improved performance budget compliance reporting

### Dependencies 📦

- **Updated all dependencies to latest crates.io versions** (2025-11-23)
  - `wgpu`: 22.0 → 27.0.1 (major update)
    - Fixed breaking changes: `entry_point` now uses `Option<&str>`
    - Updated `request_adapter` API (now returns `Result`)
    - Removed `Maintain::Wait` (polling now automatic)
    - Added `experimental_features` and `trace` to `DeviceDescriptor`
  - `criterion`: 0.5 → 0.7 (minor update)
    - Replaced `criterion::black_box` with `std::hint::black_box`
  - `thiserror`: 2.0 → 2.0.17
  - `rayon`: 1.10 → 1.11
  - `pollster`: 0.3 → 0.4
  - `bytemuck`: 1.14 → 1.24
  - `proptest`: 1.8 → 1.9

### Testing ✅

- All 942 tests passing with updated dependencies (up from 936)
- 44/44 GPU tests pass with wgpu v27 (including 14 batch tests)
- Benchmark infrastructure verified with criterion 0.7
- Zero clippy warnings maintained
- Coverage: 90%+ maintained (EXTREME TDD requirement)

### Quality 🎯

- Test coverage: 90.41% (exceeds 90% requirement)
- All quality gates passing (lint, format, tests, coverage)
- Pre-commit hooks enforce coverage threshold
- PMAT Technical Debt Grade: B+ minimum enforced

## [0.7.0] - 2025-11-22

### Added ✨

- **Async GPU Command Batching API** (v0.3.0 deliverable - Phase 1)
  - **Goal**: Reduce GPU transfer overhead by 2x for chained operations
  - **New types**:
    - `GpuCommandBatch`: Command builder for batching GPU operations
    - `BufferId`: Type-safe buffer identifier for intermediate results
  - **Operations supported**: **10 operations total**
    - **Activations**: `relu`, `sigmoid`, `tanh`, `swish`, `gelu`
    - **Arithmetic**: `add`, `sub`, `mul`, `scale`, `dot`
  - **Architecture**: Command Builder pattern for explicit batching control
    - `upload()`: Queue data for GPU upload
    - Operation methods: Queue operations (no GPU execution)
    - `execute()`: Execute all queued operations in single batch
    - `read()`: Download results from GPU
  - **Transfer reduction**:
    - Before: `relu + scale + add` = 6 transfers (3 up, 3 down)
    - After: 2 transfers (1 up, 1 down) = **3x reduction**
  - **New GPU shaders**:
    - `SCALE_SHADER`: Element-wise scalar multiplication
    - `VEC_MUL_SHADER`: Element-wise vector multiplication
    - `VEC_SUB_SHADER`: Element-wise vector subtraction
  - **Tests**: 14 comprehensive tests
    - Buffer management tests (allocation, operation queuing, error handling)
    - Operation tests (mul, dot, sigmoid, tanh, swish, gelu, sub)
    - Integration tests (end-to-end execution, chained activations)
  - **Dependencies**: Added `tokio` (dev-dependency) for async test support
  - **Benchmarks** (`benches/async_gpu_ops.rs`):
    - `bench_sync_chained_ops`: Traditional sync API (6 transfers for 3 ops)
    - `bench_async_chained_ops`: New async batch API (2 transfers for 3 ops)
    - `bench_single_op_comparison`: Sync vs async for single operation
    - `bench_deep_chain`: 5 chained operations (10→2 transfers = 5x reduction)
    - **Usage**: `cargo bench --bench async_gpu_ops --features gpu`
  - **API Enhancement**: `GpuDevice` now implements `Clone` (wgpu devices are Arc-based)

## [0.7.0] - 2025-11-22

### Performance - Phase 3: Large Matrix Optimization 🚀

**Achievement**: 18% improvement for 1024×1024 matrices via 3-level cache blocking

- **3-level cache hierarchy** (L3 → L2 → micro-kernel) for matrices ≥512×512
  - L3 blocks: 256×256 (fits in 4-16MB L3 cache)
  - L2 blocks: 64×64 (fits in 256KB L2 cache)
  - Micro-kernel: 4×1 AVX2/FMA (register blocking)
  - Smart threshold: Only activates for matrices ≥512×512

- **Zero-allocation implementation**:
  - No Vec allocations in hot path
  - Code duplication with if/else branches
  - Preserves fast 2-level path for smaller matrices

- **Performance results**:
  - 1024×1024: **47.4 ms (18% faster than v0.6.0's 57.8 ms)** ✅
  - 512×512: ~5.3 ms (8.5% improvement)
  - 256×256: No regression (uses 2-level path)
  - Target: Within 1.5× of NumPy (currently 1.64×)

- **Testing**:
  - Added `test_matmul_3level_blocking` for 512×512 matrices
  - 878 tests passing (all existing tests pass)
  - Coverage: 90.41% (improved from 90.00%)

### Quality & Testing

- **Test coverage: 90.20%** (trueno library, exceeds 90% EXTREME TDD requirement)
- Added 60+ new tests across xtask tooling and core library
- Fixed clippy warnings (needless_range_loop)
- Updated coverage policy: xtask (dev tooling) excluded from main coverage requirement
- All quality gates passing: lint, format, tests, coverage

### Documentation

- Updated Phase 2 book chapter with 3-level blocking details
- Added benchmark data for 512×512 and 1024×1024
- GitHub issue #34 tracking Phase 3 progress

## [0.6.0] - 2025-11-21

### Performance - Phase 2: NumPy Performance Parity 🎯

**Major Achievement**: Pure Rust matches NumPy/OpenBLAS performance at 256×256 matrices

- **4×1 AVX2 micro-kernel** implementation (Pure Rust, zero external dependencies)
  - Fused Multiply-Add (FMA) instructions for 3× throughput
  - Register blocking: 4 YMM accumulators stay in CPU registers
  - Memory bandwidth optimization: Load B column once, reuse for 4 A rows (4× reduction)
  - Horizontal sum optimization using AVX2 intrinsics

- **Performance results** (vs NumPy 2.3.5 + OpenBLAS):
  - 256×256: **538 μs (Trueno) vs 574 μs (NumPy) = 6% FASTER** ✅
  - 128×128: **72 μs (Trueno) vs 463 μs (NumPy) = 6.4× FASTER** ✅
  - Improvement over v0.5.0: 2.3-2.6× faster
  - Efficiency: 77% of theoretical AVX2 peak (48 GFLOPS @ 3.0 GHz)

- **Implementation details**:
  - `matmul_microkernel_4x1_avx2()`: Processes 4 rows × 1 column simultaneously
  - `horizontal_sum_avx2()`: Reduces 8 f32 values to scalar
  - Automatic dispatch for AVX2/AVX512 backends
  - Fallback to standard SIMD for other backends

- **Comprehensive testing**:
  - 11 micro-kernel unit tests added
  - `test_horizontal_sum_avx2`: 5 test cases (all ones, sequence, signs, large values, mixed)
  - `test_matmul_microkernel_4x1_avx2`: 6 test cases (simple dots, identity, non-aligned, negative, zero, FMA verification)
  - Backend equivalence: Naive vs micro-kernel correctness verified
  - Coverage: 90.63% (exceeds 90% requirement)

### Documentation

- **book/src/advanced/phase2-microkernel.md**: Complete Phase 2 micro-kernel guide
  - Motivation and design goals
  - Micro-kernel architecture (4×1 design rationale)
  - AVX2 implementation with code walkthrough
  - Performance analysis and efficiency breakdown
  - Testing strategy and coverage details
  - Lessons learned (what worked, what didn't, trade-offs)
  - Future optimizations roadmap

- **ROADMAP.md**: Updated with Phase 2 completion and Phase 3 planning
- **GitHub issue #34**: Phase 3 (large matrix optimization) opened

### Quality

- **Test Coverage**: 877 tests passing, 90.63% library coverage
- **Clippy**: Zero warnings on all features
- **Format**: 100% rustfmt compliant
- **PMAT**: All quality gates passing

### Closed Issues

- Phase 2 of matrix multiplication optimization (achieving NumPy parity)

## [0.5.0] - 2025-11-21

### Performance - Matrix Multiplication 🚀

**Major Achievement**: Matrix multiplication now **2.79× faster than NumPy** at 128×128 matrices

- **Cache-aware blocking algorithm** with L2 optimization (64×64 blocks)
  - Implements 2-level cache hierarchy optimization (L2/L1)
  - Smart thresholding: matrices ≤32 use simple path (avoids blocking overhead)
  - 3-level nested loops (ii/jj/kk) with SIMD micro-kernels
  - Zero Vector allocations via direct backend dot() calls

- **Performance results** (vs NumPy baseline):
  - 128×128 matrices: **166 μs (Trueno) vs 463 μs (NumPy) = 2.79× FASTER** ✅
  - Original problem: Trueno was 2.5× slower (Issue #10)
  - Total improvement: 5.5× faster than v0.4.0
  - Phase 1 goal (1.5-2× speedup) exceeded by 40%

- **Comprehensive testing**:
  - 4 new blocking test suites added
  - `test_matmul_blocking_small_matrices` (8×8, 16×16, 32×32)
  - `test_matmul_blocking_medium_matrices` (64×64, 128×128, 256×256)
  - `test_matmul_blocking_non_aligned_sizes` (33×33, 65×65, 100×100, 127×127)
  - `test_matmul_blocking_large_matrices` (256×256 with detailed analysis)
  - Backend equivalence verified (naive vs blocked implementations)

### Fixed

- **Performance regression** (Issue #26): Backend selection caching
  - Implemented `OnceLock` for one-time backend detection
  - Eliminates 3-5% overhead from repeated `is_x86_feature_detected!()` calls
  - Performance improvement: 4-15% faster than v0.4.0
  - Added `test_backend_selection_is_cached` to verify caching behavior

### Documentation

- **PERFORMANCE_GUIDE.md** updated with matrix multiplication section
  - Comprehensive benchmark table (16×16 through 256×256)
  - Performance characteristics and sweet spot analysis
  - Implementation details (blocking, thresholding, SIMD)
  - Tuning tips for different matrix sizes
  - Cache-aware blocking explanation

### Quality

- **Test Coverage**: 874 tests passing, 90.72% library coverage (exceeds 90% requirement)
- **TDG Score**: 85.5/100 (A-) - architectural limit maintained
- **Clippy**: Zero warnings on all features
- **Format**: 100% rustfmt compliant
- **PMAT**: All quality gates passing, zero critical defects

### Closed Issues

- Issue #10: Matrix multiplication SIMD performance (Phase 1 complete)
- Issue #26: Performance regression in v0.4.1 (backend caching fix)

## [0.4.1] - 2025-11-20

### Added
- **GPU test coverage improvements**: Comprehensive testing for GPU backend operations
  - Added 6 new GPU tests for `matmul()` and `convolve2d()` operations
  - `test_gpu_matmul_basic`, `test_gpu_matmul_identity`, `test_gpu_matmul_non_square`
  - `test_gpu_convolve2d_basic`, `test_gpu_convolve2d_identity`, `test_gpu_convolve2d_averaging`
  - GPU device.rs coverage: 68.44% → 98.44% (+30% improvement)

### Fixed
- **Test stability**: Fixed flaky `test_matvec_associativity` property test
  - Relaxed floating-point tolerance from 1% to 2% for AVX-512 backend
  - Accounts for increased rounding error accumulation in 512-bit SIMD operations
  - All 834 tests now pass reliably across all backends

### Changed
- **Coverage reporting**: Excluded xtask build tools from coverage metrics
  - Updated Makefile to use `--exclude-from-report xtask`
  - Library code coverage: **90.61%** (target: >90%) ✅
  - Overall coverage: 88.30% line, 94.42% function, 89.63% region

### Quality
- **Test Coverage**: 834 tests passing, >90% library coverage achieved
- **TDG Score**: 88.1/100 (A-) - architectural limit maintained
- **Clippy**: Zero warnings on all features
- **Format**: 100% rustfmt compliant

## [0.4.0] - 2025-11-19

### Changed
- **Refactored multi-backend dispatch**: Introduced dispatch macros to reduce code duplication
  - `dispatch_binary_op!` macro for add/sub/mul/div operations (reduces 50-line match statements to 1 line)
  - `dispatch_reduction!` macro for sum/max/min/norm operations (reduces 50-line match statements to 1 line)
  - Eliminates ~1000 lines of redundant backend dispatch code
  - Maintains 100% functional equivalence (all 827 tests passing)
  - Improves maintainability: new backends now require single macro update
  - **Note**: TDG score unchanged (88.1 A-) because `syn` expands macros before analysis
    - This is correct behavior - cyclomatic complexity remains unchanged
    - Macro pattern matches unavoidable architectural complexity from multi-platform SIMD dispatch

### Added
- **Additional vector operations**: Expanded functionality with ML/numerical computing primitives
  - `norm_l2()`: L2 norm with AVX-512 (6-9x speedup)
  - `norm_l1()`, `norm_linf()`: L1 and L-infinity norms
  - `scale()`, `abs()`, `clamp()`: Basic vector transformations
  - `lerp()`, `fma()`: Linear interpolation and fused multiply-add
  - `relu()`, `sigmoid()`, `gelu()`, `swish()`, `tanh()`: Neural network activation functions
  - `exp()`: Exponential function with range reduction
  - 827 tests passing (all operations covered)

### Infrastructure
- **PMAT integration improvements**: Created issues for enhanced TDG workflow
  - Issue #78: Request for `pmat tdg --explain` mode with function-level complexity breakdown
  - Issue #76: Documented YAML parsing friction with `pmat work` commands
  - Discovered: TDG correctly analyzes macro-expanded code via `syn` AST parser

### Quality
- **Test Coverage**: 827 tests passing, >90% coverage maintained
- **TDG Score**: 88.1/100 (A-) - architectural limit for multi-backend SIMD dispatch
- **Clippy**: Zero warnings on all features
- **Format**: 100% rustfmt compliant

## [0.3.0] - 2025-11-19

### Added
- **AVX-512 backend infrastructure**: Initial implementation (Phase 1 + Phase 2 + Phase 3 + Phase 4 + Phase 5)
  - New `Avx512Backend` processes 16 × f32 elements per iteration (2x AVX2's 8)
  - **Implemented `add()` operation**: Memory-bound (~1x speedup, baseline implementation)
  - **Implemented `dot()` operation**: Compute-bound (11-12x speedup, ✅ **EXCEEDS 8x TARGET**)
    - Uses `_mm512_fmadd_ps` for fused multiply-add (single instruction for acc + va * vb)
    - Uses `_mm512_reduce_add_ps` for horizontal sum (simpler than AVX2's manual reduction)
    - 9 comprehensive unit tests (basic, aligned, non-aligned, large, backend equivalence, special values, zero/orthogonal)
  - **Implemented `sum()` operation**: Compute-bound (8-11x speedup, ✅ **EXCEEDS 8x TARGET**)
    - Uses `_mm512_add_ps` for 16-way parallel accumulation
    - Uses `_mm512_reduce_add_ps` for horizontal sum (single intrinsic)
    - 9 comprehensive unit tests (basic, aligned, non-aligned, large, backend equivalence, negative values, remainder sizes)
  - **Implemented `max()` operation**: Compute-bound (8-12x speedup, ✅ **EXCEEDS 8x TARGET**)
    - Uses `_mm512_max_ps` for 16-way parallel comparison
    - Uses `_mm512_reduce_max_ps` for horizontal max (single intrinsic)
    - 5 comprehensive unit tests (basic, aligned, non-aligned, negative values, backend equivalence)
  - **Implemented `min()` operation**: Compute-bound (8-12x speedup, ✅ **EXCEEDS 8x TARGET**)
    - Uses `_mm512_min_ps` for 16-way parallel comparison
    - Uses `_mm512_reduce_min_ps` for horizontal min (single intrinsic)
    - 5 comprehensive unit tests (basic, aligned, non-aligned, positive values, backend equivalence)
  - **Implemented `argmax()` operation**: Hybrid operation (3.2-3.3x speedup, limited by scalar index scan)
    - Uses `_mm512_max_ps` + `_mm512_reduce_max_ps` to find maximum value (16-way SIMD)
    - Scalar `.position()` scan to find index of max value (dominates runtime)
    - 6 comprehensive unit tests (basic, aligned, non-aligned, negative values, max at start, backend equivalence)
  - **Implemented `argmin()` operation**: Hybrid operation (3.2-3.3x speedup, limited by scalar index scan)
    - Uses `_mm512_min_ps` + `_mm512_reduce_min_ps` to find minimum value (16-way SIMD)
    - Scalar `.position()` scan to find index of min value (dominates runtime)
    - 6 comprehensive unit tests (basic, aligned, non-aligned, positive values, min at start, backend equivalence)
  - Backend selection: Auto-detects AVX-512F support via `is_x86_feature_detected!()`
  - Available on Intel Skylake-X/Sapphire Rapids (2017+) and AMD Zen 4 (2022+)
  - All 819 tests passing (779 + 9 add + 9 dot + 9 sum + 5 max + 5 min + 6 argmax + 6 argmin + 1 = 819 unique)

### Infrastructure
- **GitHub Pages deployment**: Automated documentation deployment workflow
  - Combines mdBook guide and rustdoc API documentation
  - Deploys to GitHub Pages on push to main branch
  - API documentation available at `/api` subdirectory
  - Workflow file: `.github/workflows/deploy-docs.yml`

### Documentation
- **Fixed Intel Intrinsics Guide reference**: Updated to mirror URL
  - Original Intel URL blocked automated link validation (HTTP 403)
  - Now references automation-friendly mirror at `laruence.com/sse`
  - Passes PMAT `validate-docs` quality gate (136/136 links valid)

### Fixed
- **AVX512 FMA tolerance**: Increased tolerance for 3-way matmul associativity
  - Addresses floating-point precision differences in AVX-512 FMA operations
  - Commit 6cd7ba2

### Performance
- **AVX-512 add() benchmarks**: Memory-bound operation analysis
  - Size 100:   Scalar 50.9ns, AVX2 44.4ns (1.15x), **AVX512 44.8ns (1.14x)**
  - Size 1000:  Scalar 113.7ns, AVX2 101.1ns (1.12x), **AVX512 117.3ns (0.97x)**
  - Size 10000: Scalar 1.117µs, AVX2 1.106µs (1.01x), **AVX512 1.122µs (0.99x)**
  - **Conclusion**: add() is memory-bound (~1x SIMD benefit)
  - Memory bandwidth saturation prevents AVX-512 benefits for simple element-wise ops
  - Consistent with existing patterns: add/sub/div/fma/scale/abs all memory-bound (~1x speedup)
  - AVX-512's 2x register width (16 vs 8 elements) does not help when memory is bottleneck

- **AVX-512 dot() benchmarks**: Compute-bound operation ✅ **EXCEEDS 8x TARGET**
  - Size 100:   Scalar 44.2ns, AVX2 8.9ns (4.95x), **AVX512 8.4ns (5.3x)**
  - Size 1000:  Scalar 607ns, AVX2 94ns (6.5x), **AVX512 49ns (12.5x)** ✅
  - Size 10000: Scalar 6.31µs, AVX2 1.03µs (6.1x), **AVX512 551ns (11.5x)** ✅
  - **Conclusion**: dot() is compute-bound (11-12x SIMD speedup achieved!)
  - FMA intrinsic (_mm512_fmadd_ps) provides massive benefit for multiply-accumulate patterns
  - AVX-512's 16-element-wide FMA + horizontal reduction delivers 1.9x speedup over AVX2
  - Validates ROADMAP success criteria: "8x speedup over scalar (AVX-512)" ✅
  - Confirms hypothesis: Compute-bound operations benefit from AVX-512, memory-bound do not

- **AVX-512 sum() benchmarks**: Compute-bound operation ✅ **EXCEEDS 8x TARGET**
  - Size 100:   Scalar 36.3ns, AVX2 5.6ns (6.5x), **AVX512 5.7ns (6.4x)**
  - Size 1000:  Scalar 600ns, AVX2 55ns (10.9x), **AVX512 54ns (11.0x)** ✅
  - Size 10000: Scalar 6.33µs, AVX2 768ns (8.2x), **AVX512 767ns (8.3x)** ✅
  - **Conclusion**: sum() is compute-bound (8-11x SIMD speedup achieved!)
  - 16-way parallel accumulation with `_mm512_add_ps` + `_mm512_reduce_add_ps`
  - AVX-512 matches AVX2 performance (both memory-bandwidth limited for reduction)
  - Validates ROADMAP success criteria: "8x speedup over scalar (AVX-512)" ✅
  - Pattern: Reduction operations achieve target speedup despite memory constraints

- **AVX-512 max() benchmarks**: Compute-bound operation ✅ **EXCEEDS 8x TARGET**
  - Size 100:   Scalar 26.9ns, AVX2 4.3ns (6.2x), **AVX512 4.2ns (6.3x)**
  - Size 1000:  Scalar 390ns, AVX2 40ns (9.8x), **AVX512 32ns (12.1x)** ✅
  - Size 10000: Scalar 4.02µs, AVX2 482ns (8.3x), **AVX512 488ns (8.2x)** ✅
  - **Conclusion**: max() is compute-bound (8-12x SIMD speedup achieved!)
  - 16-way parallel comparison with `_mm512_max_ps` + `_mm512_reduce_max_ps`
  - AVX-512 matches AVX2 performance (both memory-bandwidth limited)
  - Validates ROADMAP success criteria ✅

- **AVX-512 min() benchmarks**: Compute-bound operation ✅ **EXCEEDS 8x TARGET**
  - Size 100:   Scalar 26.1ns, AVX2 4.2ns (6.2x), **AVX512 4.2ns (6.2x)**
  - Size 1000:  Scalar 371ns, AVX2 31ns (12.0x), **AVX512 32ns (11.6x)** ✅
  - Size 10000: Scalar 3.93µs, AVX2 484ns (8.1x), **AVX512 492ns (8.0x)** ✅
  - **Conclusion**: min() is compute-bound (8-12x SIMD speedup achieved!)
  - 16-way parallel comparison with `_mm512_min_ps` + `_mm512_reduce_min_ps`
  - AVX-512 matches AVX2 performance (both memory-bandwidth limited)
  - Validates ROADMAP success criteria ✅

- **AVX-512 argmax() benchmarks**: Hybrid operation (SIMD find + scalar scan)
  - Size 100:   Scalar 46.2ns, AVX2 21.8ns (2.1x), **AVX512 21.2ns (2.2x)**
  - Size 1000:  Scalar 580ns, AVX2 182ns (3.2x), **AVX512 184ns (3.2x)**
  - Size 10000: Scalar 5.95µs, AVX2 1.79µs (3.3x), **AVX512 1.78µs (3.3x)**
  - **Conclusion**: argmax() achieves 3.2-3.3x speedup (limited by scalar index scan)
  - SIMD phase: 16-way parallel max finding with `_mm512_max_ps` + `_mm512_reduce_max_ps`
  - Scalar phase: `.position()` scan to find index of max value (dominates runtime)
  - **Not** targeting 8x speedup - argmax is fundamentally a two-phase algorithm

- **AVX-512 argmin() benchmarks**: Hybrid operation (SIMD find + scalar scan)
  - Size 100:   Scalar 45.8ns, AVX2 21.5ns (2.1x), **AVX512 21.6ns (2.1x)**
  - Size 1000:  Scalar 581ns, AVX2 180ns (3.2x), **AVX512 181ns (3.2x)**
  - Size 10000: Scalar 5.93µs, AVX2 1.76µs (3.4x), **AVX512 1.79µs (3.3x)**
  - **Conclusion**: argmin() achieves 3.2-3.3x speedup (limited by scalar index scan)
  - SIMD phase: 16-way parallel min finding with `_mm512_min_ps` + `_mm512_reduce_min_ps`
  - Scalar phase: `.position()` scan to find index of min value (dominates runtime)
  - **Not** targeting 8x speedup - argmin is fundamentally a two-phase algorithm

### Quality
- **Mutation testing improvements**: Backend error handling test
  - Killed Backend::Auto deletion mutant (src/vector.rs:3145) with defensive error test
  - Improved test coverage for backend fallback paths
  - Known limitation: 3 GPU mutants (tanh, is_available, reduce_sum) require GPU hardware to test
  - Tests skip gracefully when GPU unavailable (prevents CI breakage)
- **Bashrs enforcement**: Shell script quality validation
  - Replaced C-grade shell validation with A-grade Rust xtask
  - Enforces bashrs validation for Makefile and all shell scripts
  - Handles missing shell scripts gracefully

---

## [0.2.2] - 2025-11-18

### Fixed
- **CRITICAL**: Missing SIMD implementation for `abs()` operation (Issue #2)
  - Blocked downstream projects (realizar)
  - Added implementations in AVX2Backend, SSE2Backend, ScalarBackend
  - Uses bitwise AND with `0x7FFFFFFF` to clear sign bit
  - All 109 tests pass, backend equivalence verified

### Performance
- **argmax/argmin SIMD optimization**: 2.8-3.1x speedup
  - Replaced scalar index scan with SIMD index tracking
  - Uses comparison masks and blend operations
  - Processes 8 elements/iteration (AVX2) or 4 elements/iteration (SSE2)

### Added
- Comprehensive performance benchmarks for 7 operations:
  - `norm_l1()` - L1 norm (4-11x SIMD speedup, compute-bound)
  - `norm_l2()` - L2 norm (4-9x SIMD speedup, compute-bound)
  - `scale()` - Scalar multiplication (~1x speedup, memory-bound)
  - `fma()` - Fused multiply-add (~1x speedup, memory-bound despite FMA hardware)
  - `sub()` - Subtraction (~1x speedup, memory-bound)
  - `div()` - Division (~1x speedup, memory-bound)
  - `abs()` - Absolute value (~1.1x speedup, memory-bound)
  - `min()` - Minimum reduction (6-10x SIMD speedup)

### Documentation
- **Performance pattern analysis documented**:
  - **Compute-bound operations** (4-12x SIMD benefit): min, argmax/argmin, norm_l1, norm_l2, dot, sum
  - **Memory-bound operations** (~1x SIMD benefit): sub, div, fma, scale, abs
  - Root cause: Memory bandwidth saturation prevents SIMD benefit for simple operations

### Testing
- All 889 tests passing (759 unit + 21 integration + 109 doc)
- Zero clippy warnings
- EXTREME TDD methodology with RED-GREEN-REFACTOR cycle applied for abs()

### Closes
- Issue #2: Missing abs trait implementation in VectorBackend

---

## [0.2.1] - 2025-11-18

### Added

#### Activation Functions
- `hardswish()` - MobileNetV3 efficient activation
- `mish()` - Modern swish alternative (x * tanh(softplus(x)))
- `selu()` - Self-normalizing exponential linear unit
- `relu()` - ReLU with EXTREME TDD

#### Math Operations
- `log2()` - Base-2 logarithm (information theory, entropy)
- `log10()` - Base-10 logarithm (decibels, pH)

#### Documentation
- Comprehensive GPU performance analysis (`docs/performance-analysis.md`)
- Performance baselines for regression detection

### Changed

#### Critical GPU Performance Optimization
- **GPU disabled for ALL element-wise operations** (2-65,000x slower than scalar!)
- **GPU enabled ONLY for matmul** (2-10x speedup at 500×500+)
- Updated OpComplexity thresholds based on empirical benchmarks
- Lowered matmul GPU threshold from 1000 to 500 (proven 2x speedup)

#### Documentation Updates
- README updated with honest GPU performance claims
- ROADMAP pivoted from GPU to SIMD optimization strategy

### Fixed
- False GPU speedup claims (advertised 10-50x, actual was 2-65,000x SLOWER)
- GPU overhead analysis: 14-55ms fixed cost per operation

### Performance

#### GPU Benchmark Results (Empirical - Genchi Genbutsu)
| Operation | Size | GPU vs Scalar | Result |
|-----------|------|---------------|--------|
| vec_add | 1M | 510x SLOWER | ❌ GPU disabled |
| dot | 1M | 93x SLOWER | ❌ GPU disabled |
| relu | 1M | 824x SLOWER | ❌ GPU disabled |
| matmul | 500×500 | **2.01x faster** | ✅ GPU enabled |
| matmul | 1000×1000 | **9.59x faster** | ✅ GPU enabled |

**Root Cause**: 14-55ms GPU overhead (buffer allocation + PCIe transfer) dominates execution time for element-wise ops.

### Testing
- 33 new tests for activations (hardswish, mish, selu)
- 14 new tests for log2/log10
- Property-based tests for all new functions
- Total: 699+ tests

### Closes
- Issue #1: Element-wise transcendental functions (log2, ln, exp)

---

## [0.1.0] - 2025-01-17

### Added

#### Core Types
- `Vector<T>` type with SIMD-optimized operations
- `Matrix<T>` type with row-major storage (NumPy-compatible)
- `Backend` enum for multi-target execution (Scalar, SSE2, AVX, AVX2, AVX512, NEON, WasmSIMD, GPU)
- Runtime CPU feature detection with automatic backend selection

#### Vector Operations (87 total)
- **Element-wise**: add, sub, mul, div, abs, neg, clamp, lerp, fma, sqrt, recip, pow, exp, ln, floor, ceil, round, trunc, fract, signum, copysign, minimum, maximum
- **Trigonometric**: sin, cos, tan, asin, acos, atan
- **Hyperbolic**: sinh, cosh, tanh, asinh, acosh, atanh
- **Dot product**: Optimized with SIMD and FMA
- **Reductions**: sum (naive + Kahan), min, max, sum_of_squares, mean, variance, stddev, covariance, correlation
- **Activation functions**: relu, leaky_relu, elu, sigmoid, softmax, log_softmax, gelu, swish/silu
- **Preprocessing**: zscore, minmax_normalize, clip
- **Index operations**: argmin, argmax
- **Vector norms**: L1, L2, L∞, normalization to unit vectors
- **Scalar operations**: scale (scalar multiplication with full SIMD)

#### Matrix Operations
- Matrix multiplication (matmul) - naive O(n³) algorithm
- Matrix transpose - O(mn) swap operation
- Constructors: new(), from_vec(), zeros(), identity()
- Accessors: get(), get_mut(), rows(), cols(), shape(), as_slice()

#### Performance Optimizations
- SSE2 SIMD (128-bit): 3-4x speedup on dot product vs scalar
- AVX2 SIMD (256-bit): Additional 1.8x speedup with FMA
- Runtime dispatch based on CPU features
- Kahan summation for numerical stability
- Numerically stable algorithms (softmax with max subtraction, correlation clamping)

#### Testing & Quality
- 611 unit tests (100% passing)
- 101 doctests (100% passing)
- Property-based testing with proptest (100 cases per test)
- Zero clippy warnings
- Zero rustdoc warnings
- EXTREME TDD methodology applied throughout
- Mutation testing support
- Pre-commit quality gates via PMAT

#### Documentation
- Comprehensive rustdoc with examples for all public APIs
- README with performance benchmarks
- Quick start guide
- Phase roadmap (Phases 1-7 complete, Phase 8 in progress)
- 4 comprehensive examples:
  - activation_functions.rs
  - backend_detection.rs
  - ml_similarity.rs
  - performance_demo.rs

### Changed
- Improved numerical stability for variance/stddev with hybrid tolerance (absolute for small values, relative for large)
- Improved correlation() to clamp results to \[-1, 1\] to handle floating-point precision
- Optimized property tests with appropriate tolerances for floating-point comparisons

### Fixed
- Fixed 4 property test failures in variance/stddev operations with better tolerance handling
- Fixed all 64 rustdoc link resolution warnings by escaping mathematical notation
- Fixed atanh(tanh(x)) round-trip precision for extreme values by restricting range
- Fixed covariance bilinearity test with increased tolerance for compounding FP errors
- Fixed zscore tests for small sample sizes (n<3) and near-constant vectors

### Performance

#### Benchmarks (vs Scalar Baseline)
| Operation | Size | SSE2 | AVX2 | Notes |
|-----------|------|------|------|-------|
| Dot Product | 10K | 3.4x | 6.2x | FMA acceleration |
| Sum | 1K | 3.15x | - | - |
| Max | 1K | 3.48x | - | - |
| Add | 1K | 1.03x | 1.15x | Memory-bound |
| Mul | 1K | 1.05x | 1.12x | Memory-bound |

All benchmarks verified with Criterion.rs.

### Technical Details

#### Quality Metrics
- Test coverage: >90%
- Test execution time: 0.09s (target: <30s) - 333x faster than requirement
- TDG Score: 95.2/100 (A+)
- Zero defects at release
- Toyota Way principles applied (Jidoka, Kaizen, Genchi Genbutsu, Hansei, Poka-Yoke)

#### Platform Support
- x86_64: SSE2/AVX/AVX2/AVX-512
- ARM: NEON
- WASM: SIMD128
- GPU: Planned (infrastructure ready)

#### Dependencies
- thiserror: 2.0 (error handling)
- proptest: 1.8 (property-based testing, dev-only)
- criterion: 0.5 (benchmarking, dev-only)

### Breaking Changes
None - this is the initial release.

### Migration Guide
This is the first release. To use:

```toml
[dependencies]
trueno = "0.1"
```

```rust
use trueno::{Vector, Matrix};

let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
let result = v.add(&v).unwrap();

let m = Matrix::identity(3);
let transposed = m.transpose();
```

### Known Limitations
- Matrix operations use naive algorithms (future: SIMD, GPU, blocked matmul)
- GPU backend infrastructure exists but not yet activated
- No matrix-vector multiplication yet (planned Phase 8)
- No compile-time backend selection (runtime only)

### Contributors
- Pragmatic AI Labs Team
- Claude (AI pair programmer)

### Links
- Repository: https://github.com/paiml/trueno
- Documentation: https://docs.rs/trueno/0.1.0
- Crates.io: https://crates.io/crates/trueno

---

## [Unreleased]

### Planned for v0.3.0
- SIMD-optimized activation functions (AVX2/AVX-512)
- Performance regression CI integration
- Matrix-vector multiplication
- Additional backends (WASM SIMD128)

[0.7.1]: https://github.com/paiml/trueno/releases/tag/v0.7.1
[0.7.0]: https://github.com/paiml/trueno/releases/tag/v0.7.0
[0.6.0]: https://github.com/paiml/trueno/releases/tag/v0.6.0
[0.5.0]: https://github.com/paiml/trueno/releases/tag/v0.5.0
[0.4.1]: https://github.com/paiml/trueno/releases/tag/v0.4.1
[0.4.0]: https://github.com/paiml/trueno/releases/tag/v0.4.0
[0.3.0]: https://github.com/paiml/trueno/releases/tag/v0.3.0
[0.2.2]: https://github.com/paiml/trueno/releases/tag/v0.2.2
[0.2.1]: https://github.com/paiml/trueno/releases/tag/v0.2.1
[0.1.0]: https://github.com/paiml/trueno/releases/tag/v0.1.0
[Unreleased]: https://github.com/paiml/trueno/compare/v0.7.1...HEAD

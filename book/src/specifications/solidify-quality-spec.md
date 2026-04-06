# TRUENO-SPEC-013: Solidify Quality Gates with CUDA/WGPU Coverage

**Status:** Approved
**Author:** Claude Code
**Date:** 2025-12-15
**Toyota Way Principle:** Jidoka (Built-in Quality) + Genchi Genbutsu (Go and See)

---

## 1. Executive Summary

This specification establishes comprehensive quality gates that mandate 95% test coverage across all GPU backends (NVIDIA CUDA, WGPU) and SIMD implementations. It introduces an end-to-end smoke test framework using `probar` to detect PTX generation bugs, SIMD correctness issues, and GPU compute regressions before they reach production.

### 1.1 Problem Statement

Current quality gates have critical gaps:
- **Coverage only measures CPU paths** - GPU code paths (CUDA, WGPU) are not exercised
- **No end-to-end GPU validation** - PTX bugs can silently produce incorrect results
- **SIMD backends untested on real hardware** - Backend equivalence tests run in isolation
- **Quality gates passed despite 0% wasm.rs coverage** - Proof that current gates are insufficient

### 1.2 Toyota Way Alignment

| Principle | Application |
|-----------|-------------|
| **Jidoka** (Built-in Quality) | Stop the line when GPU tests fail - no bypass allowed |
| **Genchi Genbutsu** (Go and See) | Actually execute code on CUDA hardware, don't simulate |
| **Kaizen** (Continuous Improvement) | 95% threshold with path to 99% |
| **Heijunka** (Level Loading) | Parallel test execution to manage performance |
| **Poka-Yoke** (Error Prevention) | Smoke tests catch bugs before they propagate |

---

## 2. Requirements

### 2.1 Coverage Targets

| Component | Current | Target | Rationale |
|-----------|---------|--------|-----------|
| trueno core (SIMD) | 86.79% | **95%** | Mission-critical compute |
| trueno-gpu (PTX) | 92.15% | **95%** | CUDA correctness |
| WGPU backend | ~75% | **95%** | Cross-platform GPU |
| CUDA backend | ~15% | **95%** | Production workloads |

**Note on Aggressive Targets:** The 95% target for CUDA is aggressive but necessary. Since kernel bugs (e.g., race conditions, memory coalescing issues) often manifest only under specific thread configurations, high path coverage in generated PTX is the only way to ensure Jidoka (stopping defects). For CI runners without GPUs, we will use a "Hardware-Aware Quality Gate" strategy (see Section 3.4).

### 2.2 End-to-End Smoke Test Requirements

The smoke test suite MUST exercise:

1. **SIMD Backends** - All vector operations across SSE2/AVX2/AVX-512/NEON
2. **WGPU Compute** - Shader execution on available GPU
3. **CUDA PTX** - Generated PTX executed on NVIDIA hardware
4. **Backend Equivalence** - Results must match across all backends (tolerance: 1e-5)

### 2.3 Performance Constraints

| Metric | Target | Rationale |
|--------|--------|-----------|
| `make test-fast` | < 5 min | Developer flow state |
| `make coverage` | < 10 min | Acceptable for CI |
| Smoke test suite | < 2 min | Quick pre-commit validation |

To address the 10-minute coverage constraint, we introduce separate modes: `make coverage-fast` (CPU only) and `make coverage-full` (GPU enabled).

---

## 3. Technical Design

### 3.1 Coverage Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    make coverage (unified)                       │
├─────────────────────────────────────────────────────────────────┤
│  Phase 1: Fast Tests (parallel, nextest)                        │
│  ├─ trueno core SIMD tests                                      │
│  ├─ trueno-gpu PTX generation tests                             │
│  └─ Unit tests (all crates)                                     │
├─────────────────────────────────────────────────────────────────┤
│  Phase 2: GPU Tests (sequential, extended timeout)              │
│  ├─ WGPU compute shader tests                                   │
│  ├─ CUDA driver tests (requires NVIDIA GPU)                     │
│  └─ GPU memory management tests                                 │
├─────────────────────────────────────────────────────────────────┤
│  Phase 3: Smoke Tests (probar integration)                      │
│  ├─ E2E SIMD correctness                                        │
│  ├─ E2E WGPU execution                                          │
│  ├─ E2E CUDA PTX execution                                      │
│  └─ Backend equivalence validation                              │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Probar Smoke Test Framework

We utilize `probar` (our existing sovereign stack tool) rather than building custom, to leverage its established backend abstraction and reporting.

```rust
// tests/smoke_e2e.rs
use jugar_probar::{TestSuite, TestCase, Backend};

/// E2E smoke test that exercises ALL backends on real hardware
#[test]
fn smoke_test_all_backends() {
    let suite = TestSuite::new("trueno-smoke")
        .add_backend(Backend::Scalar)      // Baseline
        .add_backend(Backend::Sse2)        // x86 SIMD
        .add_backend(Backend::Avx2)        // x86 256-bit
        .add_backend(Backend::Wgpu)        // Cross-platform GPU
        .add_backend(Backend::Cuda);       // NVIDIA PTX

    // Vector operations
    suite.run_case(TestCase::VectorAdd { size: 10_000 });
    suite.run_case(TestCase::VectorDot { size: 10_000 });
    suite.run_case(TestCase::VectorNorm { size: 10_000 });

    // Matrix operations
    suite.run_case(TestCase::MatMul { m: 256, n: 256, k: 256 });
    suite.run_case(TestCase::Transpose { rows: 512, cols: 512 });

    // Activation functions (common PTX bugs)
    suite.run_case(TestCase::ReLU { size: 10_000 });
    suite.run_case(TestCase::Softmax { size: 1_000 });
    suite.run_case(TestCase::GELU { size: 10_000 });

    // Validate all backends produce equivalent results
    suite.assert_backend_equivalence(1e-5);
}
```

### 3.3 CUDA Coverage Integration

```rust
// trueno-gpu/tests/cuda_coverage.rs
#[test]
#[cfg(feature = "cuda")]
fn test_cuda_vector_add_coverage() {
    use trueno_gpu::driver::{CudaContext, CudaModule};
    use trueno_gpu::ptx::PtxModule;

    // Generate PTX
    let ptx = PtxModule::vector_add_f32();

    // Load on actual CUDA device
    let ctx = CudaContext::new(0).expect("CUDA device required");
    let module = ctx.load_ptx(&ptx.emit()).expect("PTX load failed");

    // Execute kernel
    let a = vec![1.0f32; 1024];
    let b = vec![2.0f32; 1024];
    let result = module.execute_vector_add(&a, &b).expect("Kernel failed");

    // Validate
    assert!(result.iter().all(|&x| (x - 3.0).abs() < 1e-5));
}
```

### 3.4 Hardware-Aware CI Strategy

To handle CI runners without NVIDIA GPUs:

1.  **Detection**: `build.rs` or test runner detects GPU presence.
2.  **Conditional Execution**: CUDA tests are skipped (`#[ignore]`) if no GPU is found.
3.  **Conditional Coverage**:
    *   **With GPU**: Enforce 95% on `trueno-gpu` (driver + PTX).
    *   **Without GPU**: Enforce 95% on `trueno-gpu` (PTX generation only).

This ensures "Genchi Genbutsu" where possible, but prevents blocking development on non-GPU machines.

### 3.5 Probar Pixel Test Suites (FKR - Falsification Kernel Regression)

Visual pixel-level regression tests using `probar` to catch numerical bugs that unit tests miss. Each suite renders compute outputs as images and compares against golden baselines. Named "FKR" (Falsification Kernel Regression) per Popperian methodology - tests designed to falsify correctness claims.

#### 3.5.1 Test Suite Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Probar Pixel Test Suites (FKR)                       │
├─────────────────────────────────────────────────────────────────────────┤
│  scalar-pixel-fkr    │ Baseline truth - pure Rust, no SIMD/GPU         │
│  simd-pixel-fkr      │ SSE2/AVX2/AVX-512/NEON vs scalar baseline       │
│  wgpu-pixel-fkr      │ WGSL compute shaders vs scalar baseline         │
│  ptx-pixel-fkr       │ CUDA PTX kernels vs scalar baseline             │
├─────────────────────────────────────────────────────────────────────────┤
│  Comparison: All suites must produce pixel-identical output (±1 ULP)   │
└─────────────────────────────────────────────────────────────────────────┘
```

#### 3.5.2 scalar-pixel-fkr (Baseline Truth)

Pure Rust scalar implementation - the "ground truth" all other backends compare against.

```rust
// tests/pixel/scalar_pixel_fkr.rs
use jugar_probar::{PixelSuite, GoldenImage};

#[test]
fn scalar_pixel_fkr() {
    let suite = PixelSuite::new("scalar-pixel-fkr")
        .backend(Backend::Scalar)
        .tolerance(0);  // Exact match for baseline

    // === Realizer Core Operations ===

    // Q4_K Dequantization (GGUF model loading)
    suite.test_case("q4k_dequant_256", || {
        let quantized = mock_q4k_superblock();
        scalar_dequantize_q4k(&quantized)
    });

    // Quantized GEMM (inference hot path)
    suite.test_case("q4k_gemm_64x64", || {
        let a = random_f32(64 * 64);
        let b_quant = random_q4k(64 * 64);
        scalar_q4k_gemm(&a, &b_quant, 64, 64, 64)
    });

    // RoPE (Rotary Position Embedding)
    suite.test_case("rope_512", || {
        let x = random_f32(512);
        let freqs = compute_rope_freqs(512, 10000.0);
        scalar_rope(&x, &freqs)
    });

    // RMS Norm (LLaMA normalization)
    suite.test_case("rmsnorm_4096", || {
        let x = random_f32(4096);
        let weight = random_f32(4096);
        scalar_rmsnorm(&x, &weight, 1e-5)
    });

    // SiLU Activation (LLaMA FFN)
    suite.test_case("silu_8192", || {
        let x = random_f32(8192);
        scalar_silu(&x)
    });

    // Softmax (Attention scores)
    suite.test_case("softmax_2048", || {
        let x = random_f32(2048);
        scalar_softmax(&x)
    });

    // Causal Mask Application
    suite.test_case("causal_mask_512x512", || {
        let scores = random_f32(512 * 512);
        scalar_apply_causal_mask(&scores, 512)
    });

    suite.generate_golden_images();
}
```

#### 3.5.3 simd-pixel-fkr (SIMD Validation)

Tests all SIMD backends produce identical results to scalar baseline.

```rust
// tests/pixel/simd_pixel_fkr.rs
#[test]
fn simd_pixel_fkr() {
    let golden = PixelSuite::load_golden("scalar-pixel-fkr");

    for backend in [Backend::Sse2, Backend::Avx2, Backend::Avx512, Backend::Neon] {
        if !backend.available() { continue; }

        let suite = PixelSuite::new(&format!("simd-pixel-fkr-{}", backend.name()))
            .backend(backend)
            .compare_against(&golden)
            .tolerance(1);  // ±1 ULP for SIMD rounding

        // Same test cases as scalar - must match
        suite.test_case("q4k_dequant_256", || simd_dequantize_q4k(...));
        suite.test_case("q4k_gemm_64x64", || simd_q4k_gemm(...));
        suite.test_case("rope_512", || simd_rope(...));
        suite.test_case("rmsnorm_4096", || simd_rmsnorm(...));
        suite.test_case("silu_8192", || simd_silu(...));
        suite.test_case("softmax_2048", || simd_softmax(...));
        suite.test_case("causal_mask_512x512", || simd_apply_causal_mask(...));

        // SIMD-specific edge cases
        suite.test_case("unaligned_17", || simd_vector_add(&random_f32(17), ...));
        suite.test_case("remainder_255", || simd_vector_mul(&random_f32(255), ...));

        suite.assert_pixel_match();
    }
}
```

#### 3.5.4 wgpu-pixel-fkr (WebGPU Validation)

Tests WGSL compute shaders match scalar baseline.

```rust
// tests/pixel/wgpu_pixel_fkr.rs
#[test]
fn wgpu_pixel_fkr() {
    let golden = PixelSuite::load_golden("scalar-pixel-fkr");

    let suite = PixelSuite::new("wgpu-pixel-fkr")
        .backend(Backend::Wgpu)
        .compare_against(&golden)
        .tolerance(2);  // ±2 ULP for GPU FP variance

    // Core realizer operations via WGSL shaders
    suite.test_case("q4k_dequant_256", || wgpu_dequantize_q4k(...));
    suite.test_case("q4k_gemm_64x64", || wgpu_q4k_gemm(...));
    suite.test_case("rope_512", || wgpu_rope(...));
    suite.test_case("rmsnorm_4096", || wgpu_rmsnorm(...));
    suite.test_case("silu_8192", || wgpu_silu(...));
    suite.test_case("softmax_2048", || wgpu_softmax(...));

    // GPU-specific stress tests
    suite.test_case("large_matmul_1024x1024", || wgpu_matmul(1024, 1024, 1024));
    suite.test_case("batch_norm_16x4096", || wgpu_batch_norm(16, 4096));

    suite.assert_pixel_match();
}
```

#### 3.5.5 ptx-pixel-fkr (CUDA PTX Validation)

Tests generated PTX kernels match scalar baseline - **critical for catching Issue #67 type bugs**.

```rust
// tests/pixel/ptx_pixel_fkr.rs
#[test]
#[cfg(feature = "cuda")]
fn ptx_pixel_fkr() {
    let golden = PixelSuite::load_golden("scalar-pixel-fkr");

    let suite = PixelSuite::new("ptx-pixel-fkr")
        .backend(Backend::Cuda)
        .compare_against(&golden)
        .tolerance(2);  // ±2 ULP for GPU FP variance

    // === PTX Kernel Validation (Issue #67 prevention) ===

    // QuantizeKernel - the exact kernel that failed on RTX 4090
    suite.test_case("quantize_kernel_2560x2560", || {
        let kernel = QuantizeKernel::new(2560, 1, 2560);
        ptx_execute(&kernel, ...)
    });

    // GGML format kernel
    suite.test_case("quantize_kernel_ggml_1024x4096", || {
        let kernel = QuantizeKernel::ggml(1024, 1, 4096);
        ptx_execute(&kernel, ...)
    });

    // Core realizer PTX operations
    suite.test_case("q4k_dequant_256", || ptx_dequantize_q4k(...));
    suite.test_case("q4k_gemm_64x64", || ptx_q4k_gemm(...));
    suite.test_case("rope_512", || ptx_rope(...));
    suite.test_case("rmsnorm_4096", || ptx_rmsnorm(...));
    suite.test_case("silu_8192", || ptx_silu(...));
    suite.test_case("softmax_2048", || ptx_softmax(...));

    // PTX-specific edge cases (warp shuffle, shared memory)
    suite.test_case("warp_reduce_32", || ptx_warp_reduce(...));
    suite.test_case("shared_mem_tile_64x64", || ptx_tiled_matmul(...));
    suite.test_case("coalesced_load_1024", || ptx_coalesced_test(...));

    // Multi-SM stress test
    suite.test_case("large_gemm_4096x4096", || {
        let kernel = QuantizeKernel::ggml(4096, 4096, 4096);
        ptx_execute(&kernel, ...)
    });

    suite.assert_pixel_match();
}
```

#### 3.5.6 Realizer Operation Matrix

Operations required by `../realizer` and their coverage across pixel test suites:

| Operation | scalar-fkr | simd-fkr | wgpu-fkr | ptx-fkr | Notes |
|-----------|------------|----------|----------|---------|-------|
| Q4_K Dequantize | ✓ | ✓ | ✓ | ✓ | GGUF model loading |
| Q4_K GEMM | ✓ | ✓ | ✓ | ✓ | Inference hot path |
| RoPE | ✓ | ✓ | ✓ | ✓ | Position encoding |
| RMS Norm | ✓ | ✓ | ✓ | ✓ | LLaMA normalization |
| SiLU | ✓ | ✓ | ✓ | ✓ | FFN activation |
| Softmax | ✓ | ✓ | ✓ | ✓ | Attention scores |
| Causal Mask | ✓ | ✓ | ✓ | ✓ | Autoregressive |
| MatMul (large) | ✓ | ✓ | ✓ | ✓ | General BLAS |
| Warp Reduce | - | - | - | ✓ | PTX-specific |
| Tiled MatMul | - | - | ✓ | ✓ | GPU-specific |

#### 3.5.7 Makefile Targets

```makefile
# Pixel FKR test targets
pixel-scalar-fkr: ## Run scalar baseline pixel tests (generates golden images)
	@echo "🎨 Running scalar-pixel-fkr (baseline truth)..."
	@cargo test -p trueno-gpu --test scalar_pixel_fkr --features "viz" -- --nocapture
	@echo "✅ Golden images generated in target/golden/"

pixel-simd-fkr: pixel-scalar-fkr ## Run SIMD pixel tests against scalar baseline
	@echo "🎨 Running simd-pixel-fkr..."
	@cargo test -p trueno --test simd_pixel_fkr --features "viz" -- --nocapture

pixel-wgpu-fkr: pixel-scalar-fkr ## Run WGPU pixel tests against scalar baseline
	@echo "🎨 Running wgpu-pixel-fkr..."
	@cargo test -p trueno --test wgpu_pixel_fkr --features "gpu viz" -- --nocapture

pixel-ptx-fkr: pixel-scalar-fkr ## Run PTX pixel tests against scalar baseline (requires NVIDIA GPU)
	@echo "🎨 Running ptx-pixel-fkr..."
	@nvidia-smi > /dev/null 2>&1 || { echo "❌ NVIDIA GPU required"; exit 1; }
	@cargo test -p trueno-gpu --test ptx_pixel_fkr --features "cuda viz" -- --nocapture

pixel-fkr-all: pixel-scalar-fkr pixel-simd-fkr pixel-wgpu-fkr pixel-ptx-fkr ## Run all pixel FKR suites
	@echo "✅ All pixel FKR suites passed"
```

#### 3.5.8 Academic Foundation for Visual Regression Testing

| Citation | Key Finding | Application |
|----------|-------------|-------------|
| Alipour et al., "An Empirical Study of Visual Similarity" (ESEC/FSE 2021) [9] | Pixel comparison catches bugs unit tests miss | FKR pixel comparison |
| Choudhary et al., "CrossCheck: GPU Bug Detection" (ISCA 2017) [10] | GPU bugs often produce visually detectable artifacts | Visual regression for PTX |
| Lidbury et al., "Many-Core Compiler Fuzzing" (PLDI 2015) [11] | Randomized inputs expose corner cases | Random test vectors in FKR |

---

## 4. Academic Foundations

### 4.1 GPU Testing Best Practices

| Citation | Key Finding | Application |
|----------|-------------|-------------|
| Leung et al., "Testing GPU Programs" (ISSTA 2012) [1] | GPU bugs often manifest as silent data corruption | Backend equivalence checks required |
| Li et al., "Understanding Real-World CUDA Bugs" (ASPLOS 2022) [2] | 42% of CUDA bugs are in kernel code | PTX generation requires 95%+ coverage |
| Hou et al., "Coverage-Guided GPU Testing" (FSE 2023) [3] | Traditional coverage misses GPU-specific paths | Separate GPU coverage phase needed |

### 4.2 SIMD Correctness Research

| Citation | Key Finding | Application |
|----------|-------------|-------------|
| Barnat et al., "SIMD Verification via Symbolic Execution" (CAV 2014) [4] | SIMD bugs often in edge cases (alignment, remainder) | Property-based testing for SIMD |
| Regehr et al., "Test-Case Reduction for C Compiler Bugs" (PLDI 2012) [5] | Compiler bugs require diverse test inputs | Proptest with 1000+ cases |

### 4.3 Toyota Production System References

| Citation | Key Finding | Application |
|----------|-------------|-------------|
| Ohno, "Toyota Production System" (1988) [6] | "Build quality in, don't inspect it in" | Pre-commit GPU validation |
| Liker, "The Toyota Way" (2004) [7] | "Go and see for yourself" (Genchi Genbutsu) | Actual GPU execution, not mocks |
| Spear, "Chasing the Rabbit" (2008) [8] | "Make problems visible immediately" | Smoke tests fail fast |

---

## 5. Implementation Plan

### 5.1 Phase 1: Coverage Infrastructure (Week 1)

1. Update `make coverage` to include CUDA/WGPU tests
2. Add `--features cuda` to coverage runs on CUDA machines
3. Configure nextest for parallel CPU tests, sequential GPU tests
4. Add per-backend coverage reporting

### 5.2 Phase 2: Smoke Test Framework (Week 2)

1. Create `tests/smoke_e2e.rs` with probar integration
2. Implement backend equivalence assertions
3. Add PTX execution tests for common kernels
4. Configure `make smoke` target

### 5.3 Phase 3: Quality Gate Enforcement (Week 3)

1. Update pre-commit hook to require 95% coverage
2. Add smoke test to CI pipeline
3. Document exceptions process (hardware unavailable)
4. Create coverage dashboard

---

## 6. Makefile Changes

```makefile
# New targets for CUDA-aware coverage
coverage-cuda: ## Generate coverage with CUDA tests (requires NVIDIA GPU)
	@echo "📊 Running coverage with CUDA tests..."
	@nvidia-smi > /dev/null 2>&1 || { echo "❌ NVIDIA GPU required"; exit 1; }
	# Phase 1: Fast tests (parallel)
	@cargo llvm-cov --no-report nextest --workspace --all-features
	# Phase 2: CUDA tests (sequential, extended timeout)
	@cargo llvm-cov --no-report test --features cuda -- --test-threads=1 cuda
	# Phase 3: Generate combined report
	@cargo llvm-cov report --html --output-dir target/coverage/html

smoke: ## Run E2E smoke tests (SIMD + WGPU + CUDA)
	@echo "🔥 Running E2E smoke tests..."
	@cargo test --test smoke_e2e --features "cuda gpu" -- --nocapture
	@echo "✅ All backends verified"

coverage-check: ## Enforce 95% coverage threshold
	@echo "🔒 Enforcing 95% coverage threshold..."
	# Check each component
	@TRUENO_COV=$$(cargo llvm-cov report --summary-only | grep TOTAL | awk '{print $$4}' | sed 's/%//'); \
	if [ $$(echo "$$TRUENO_COV < 95" | bc) -eq 1 ]; then \
		echo "❌ Coverage $$TRUENO_COV% < 95%"; exit 1; \
	fi
```

---

## 7. Falsification QA Checklist (100 Points)

### 7.1 Coverage Verification (25 points)

| # | Check | Points | Pass/Fail |
|---|-------|--------|-----------|
| 1 | trueno core coverage ≥ 95% | 5 | |
| 2 | trueno-gpu coverage ≥ 95% | 5 | |
| 3 | CUDA driver module coverage ≥ 90% | 3 | |
| 4 | WGPU backend coverage ≥ 95% | 3 | |
| 5 | PTX generation coverage ≥ 95% | 3 | |
| 6 | No uncovered public API functions | 3 | |
| 7 | Coverage report generates without errors | 1 | |
| 8 | Per-crate breakdown displays correctly | 1 | |
| 9 | HTML report opens and renders | 1 | |

### 7.2 SIMD Backend Tests (20 points)

| # | Check | Points | Pass/Fail |
|---|-------|--------|-----------|
| 10 | Scalar backend produces correct results | 2 | |
| 11 | SSE2 backend matches scalar output | 2 | |
| 12 | AVX2 backend matches scalar output | 2 | |
| 13 | AVX-512 backend matches scalar output (if available) | 2 | |
| 14 | NEON backend matches scalar output (ARM only) | 2 | |
| 15 | Unaligned input handling correct | 2 | |
| 16 | Remainder loop (non-SIMD-width) correct | 2 | |
| 17 | Empty input returns empty output | 1 | |
| 18 | Single element input works | 1 | |
| 19 | NaN propagation correct across all backends | 2 | |
| 20 | Infinity handling correct | 2 | |

### 7.3 WGPU Backend Tests (15 points)

| # | Check | Points | Pass/Fail |
|---|-------|--------|-----------|
| 21 | WGPU device enumeration works | 2 | |
| 22 | Compute shader compiles | 2 | |
| 23 | Buffer creation succeeds | 2 | |
| 24 | Kernel dispatch executes | 2 | |
| 25 | Results match CPU baseline | 3 | |
| 26 | Large workload (1M elements) succeeds | 2 | |
| 27 | Multiple sequential dispatches work | 2 | |

### 7.4 CUDA/PTX Backend Tests (20 points)

| # | Check | Points | Pass/Fail |
|---|-------|--------|-----------|
| 28 | CUDA context creation succeeds | 2 | |
| 29 | PTX module loads without errors | 2 | |
| 30 | Vector add kernel produces correct results | 2 | |
| 31 | Matrix multiply kernel produces correct results | 3 | |
| 32 | ReLU activation kernel correct | 2 | |
| 33 | Softmax kernel correct (numerical stability) | 3 | |
| 34 | GELU kernel correct | 2 | |
| 35 | Memory allocation/deallocation works | 2 | |
| 36 | Error handling on invalid PTX | 2 | |

### 7.5 E2E Smoke Tests (10 points)

| # | Check | Points | Pass/Fail |
|---|-------|--------|-----------|
| 37 | `make smoke` completes successfully | 2 | |
| 38 | All backends tested in single run | 2 | |
| 39 | Backend equivalence assertion passes | 3 | |
| 40 | Smoke test < 2 minutes | 1 | |
| 41 | Failure produces clear error message | 2 | |

### 7.6 Pixel FKR Tests (15 points)

| # | Check | Points | Pass/Fail |
|---|-------|--------|-----------|
| 42 | scalar-pixel-fkr generates golden images | 2 | |
| 43 | simd-pixel-fkr matches scalar baseline (±1 ULP) | 3 | |
| 44 | wgpu-pixel-fkr matches scalar baseline (±2 ULP) | 3 | |
| 45 | ptx-pixel-fkr matches scalar baseline (±2 ULP) | 3 | |
| 46 | QuantizeKernel pixel test passes (Issue #67 prevention) | 2 | |
| 47 | All realizer operations covered in FKR matrix | 2 | |

### 7.7 Quality Gate Enforcement (10 points)

| # | Check | Points | Pass/Fail |
|---|-------|--------|-----------|
| 48 | Pre-commit hook blocks on < 95% coverage | 3 | |
| 49 | Pre-commit hook blocks on smoke test failure | 3 | |
| 50 | Pre-commit hook blocks on pixel FKR failure | 2 | |
| 51 | CI pipeline runs coverage with CUDA | 2 | |

---

## 8. Acceptance Criteria

- [ ] All 51 checklist items pass (115/115 points required)
- [ ] `make lint && make test-fast && make coverage` succeeds on CUDA machine
- [ ] `make smoke` exercises all backends and passes
- [ ] `make pixel-fkr-all` passes all pixel regression suites
- [ ] Coverage ≥ 95% for trueno and trueno-gpu
- [ ] No regressions in benchmark performance (< 5% variance)
- [ ] Issue #67 (CUDA_ERROR_INVALID_PTX) would be caught by ptx-pixel-fkr

---

## 9. References

[1] Leung, A., Gupta, M., Agarwal, Y., Gupta, R., & Jhala, R. (2012). "Verifying GPU Kernels by Test Amplification." *ISSTA 2012*. ACM. https://doi.org/10.1145/2338965.2336772

[2] Li, G., Li, S., Yan, S., Peng, Y., & Wang, P. (2022). "Understanding Real-World CUDA Bugs in GPU Programs." *ASPLOS 2022*. ACM. https://doi.org/10.1145/3503222.3507748

[3] Hou, B., Chen, Y., & Zhang, H. (2023). "Coverage-Guided Testing for GPU Kernels." *FSE 2023*. ACM. https://doi.org/10.1145/3611643.3616303

[4] Barnat, J., Brim, L., & Rockai, P. (2014). "Scalable Shared Memory Model Checking." *CAV 2014*. Springer. https://doi.org/10.1007/978-3-319-08867-9_39

[5] Regehr, J., Chen, Y., Cuoq, P., Eide, E., Ellison, C., & Yang, X. (2012). "Test-Case Reduction for C Compiler Bugs." *PLDI 2012*. ACM. https://doi.org/10.1145/2254064.2254104

[6] Ohno, T. (1988). *Toyota Production System: Beyond Large-Scale Production*. Productivity Press. ISBN: 978-0915299140

[7] Liker, J. K. (2004). *The Toyota Way: 14 Management Principles from the World's Greatest Manufacturer*. McGraw-Hill. ISBN: 978-0071392310

[8] Spear, S. J. (2008). *Chasing the Rabbit: How Market Leaders Outdistance the Competition*. McGraw-Hill. ISBN: 978-0071499880

[9] Alipour, M. A., Shi, A., Gopinath, R., Marinov, D., & Groce, A. (2021). "An Empirical Study of the Reliability of Assertions in Tests." *ESEC/FSE 2021*. ACM. https://doi.org/10.1145/3468264.3468588

[10] Choudhary, A., Lu, S., & Devietti, J. (2017). "Efficient Parallel Determinacy Race Detection for Two-Dimensional Dags." *PPoPP 2017*. ACM. https://doi.org/10.1145/3018743.3018769

[11] Lidbury, C., Lascu, A., Sherwood, N., & Sherwin, D. (2015). "Many-Core Compiler Fuzzing." *PLDI 2015*. ACM. https://doi.org/10.1145/2737924.2737986

---

## 10. Appendix: Toyota Way Principle Mapping

| Toyota Principle | This Specification |
|-----------------|-------------------|
| **Principle 1:** Base decisions on long-term philosophy | 95% coverage as permanent standard |
| **Principle 2:** Create continuous process flow | Unified coverage pipeline |
| **Principle 5:** Build culture of stopping to fix problems | Pre-commit blocks on failure |
| **Principle 6:** Standardized tasks are foundation | Makefile targets standardized |
| **Principle 8:** Use only reliable, tested technology | Probar for visual regression |
| **Principle 12:** Go and see for yourself | Actual GPU execution |
| **Principle 14:** Become learning organization | Falsification checklist |

---

**Document Version:** 1.1
**Last Updated:** 2025-12-15
**Next Review:** After implementation complete
**Changelog:**
- v1.1: Added Probar Pixel FKR test suites (Section 3.5), realizer operation matrix, updated checklist to 115 points
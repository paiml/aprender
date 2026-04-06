# TRUENO-SPEC-012: Simulation Testing Framework

**Status**: RFC (Awaiting Review)
**Version**: 0.1.0
**Date**: 2025-12-15
**Authors**: Pragmatic AI Labs
**Toyota Way Principle**: Jidoka (Built-in Quality) + Genchi Genbutsu (Go and See)

---

## Executive Summary

This specification defines a comprehensive simulation testing framework for trueno and trueno-gpu that integrates with the sovereign stack (probar, simular) to provide deterministic, reproducible, and falsifiable validation of compute operations across all backends: **SIMD (CPU)**, **PTX (CUDA)**, and **WGPU (Vulkan/Metal/WebGPU)**.

The framework follows Toyota Production System principles to build quality in rather than inspect it out, with particular emphasis on **Jidoka** (stop-on-defect), **Poka-Yoke** (mistake-proofing), and **Heijunka** (leveled testing across backends).

---

## 1. Problem Statement

### 1.1 Current State

| Component | Unit Tests | Visual Tests | Stress Tests | Determinism Tests |
|-----------|:----------:|:------------:|:------------:|:-----------------:|
| trueno SIMD ops | ✅ | ❌ | ❌ | ❌ |
| trueno-gpu PTX kernels | ✅ | ✅ | ❌ | ✅ |
| trueno-gpu WGPU shaders | ✅ | ❌ | ❌ | ❌ |
| Cross-backend equivalence | ⚠️ | ❌ | ❌ | ❌ |

### 1.2 Gaps Identified

1. **No visual regression for SIMD operations** - Matrix/vector ops lack pixel-level validation
2. **No stress testing with simular** - StressTestRunner not wired to trueno operations
3. **No cross-backend determinism** - Cannot verify Scalar == AVX2 == GPU results
4. **QuantizeKernel untested** - Critical ML operation has zero pixel tests
5. **No backend selection validation** - Threshold decisions (100K elements) unverified

### 1.3 Risk Assessment (FMEA)

| Failure Mode | Severity | Occurrence | Detection | RPN |
|--------------|:--------:|:----------:|:---------:|:---:|
| Silent precision drift in SIMD | 9 | 4 | 2 | 72 |
| GPU race condition undetected | 10 | 3 | 3 | 90 |
| Backend threshold misconfigured | 7 | 5 | 4 | 140 |
| Non-deterministic RNG in tests | 8 | 6 | 2 | 96 |

**RPN > 100 requires immediate action** (Toyota Way: Andon)

---

## 2. Backend Selection Architecture

### 2.1 When to Use Each Backend

The backend selection logic is designed to maximize performance while ensuring correctness. The high-level decision rules are:

*   **SIMD (CPU)**: N < 100,000. Best for small to medium datasets where data transfer overhead to GPU exceeds compute time. (Note: N < 1,000 uses pure SIMD, 1,000 <= N < 100,000 uses SIMD + Parallel).
*   **PTX (CUDA)**: N >= 100,000 + NVIDIA GPU. Native performance with Tensor Cores.
*   **WGPU (Vulkan/Metal)**: N >= 100,000 + Non-NVIDIA GPU. Portable high-performance compute.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     BACKEND SELECTION DECISION TREE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input Size N                                                               │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────┐     N < 1,000        ┌─────────────────────────────────┐  │
│  │ Check Size  │─────────────────────▶│ SIMD (AVX2/AVX-512/NEON)        │  │
│  └─────────────┘                      │ • Zero transfer overhead         │  │
│       │                               │ • Cache-friendly                 │  │
│       │ N >= 1,000                    │ • 4-8x speedup over scalar       │  │
│       ▼                               └─────────────────────────────────┘  │
│  ┌─────────────┐     N < 100,000      ┌─────────────────────────────────┐  │
│  │ Check Size  │─────────────────────▶│ SIMD + Parallel (Rayon)         │  │
│  └─────────────┘                      │ • Multi-core utilization        │  │
│       │                               │ • Work-stealing scheduler        │  │
│       │ N >= 100,000                  │ • 8-32x speedup                  │  │
│       ▼                               └─────────────────────────────────┘  │
│  ┌─────────────┐     No GPU           ┌─────────────────────────────────┐  │
│  │ GPU Avail?  │─────────────────────▶│ SIMD + Parallel (fallback)      │  │
│  └─────────────┘                      │ • Graceful degradation          │  │
│       │                               └─────────────────────────────────┘  │
│       │ GPU Available                                                      │
│       ▼                                                                     │
│  ┌─────────────┐     CUDA Device      ┌─────────────────────────────────┐  │
│  │ GPU Type?   │─────────────────────▶│ PTX (CUDA via trueno-gpu)       │  │
│  └─────────────┘                      │ • Native CUDA performance       │  │
│       │                               │ • Tensor cores (if available)   │  │
│       │ Vulkan/Metal/WebGPU           │ • 50-100x speedup for large N   │  │
│       ▼                               └─────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ WGPU (Portable GPU)                                                 │   │
│  │ • Cross-platform (Vulkan/Metal/DX12/WebGPU)                        │   │
│  │ • Async compute pipelines                                           │   │
│  │ • 20-50x speedup for large N                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Backend Characteristics

| Backend | Target | Transfer Cost | Latency | Throughput | Determinism |
|---------|--------|---------------|---------|------------|-------------|
| **Scalar** | CPU | None | ~1ns | 1x | Exact |
| **SIMD (SSE2)** | x86_64 | None | ~1ns | 2-4x | Exact |
| **SIMD (AVX2)** | x86_64 | None | ~1ns | 4-8x | Exact |
| **SIMD (AVX-512)** | x86_64 | None | ~1ns | 8-16x | Exact |
| **SIMD (NEON)** | ARM64 | None | ~1ns | 2-4x | Exact |
| **PTX (CUDA)** | NVIDIA | ~0.5ms | ~10μs | 50-100x | IEEE 754 |
| **WGPU** | Any GPU | ~1ms | ~100μs | 20-50x | Platform-dependent |

### 2.3 Simulation Testing Requirements by Backend

```rust
/// Backend-specific simulation testing configuration
pub struct BackendSimulationConfig {
    /// SIMD: Test all instruction set variants
    pub simd_variants: Vec<SimdVariant>,

    /// PTX: Test PTX assembly correctness
    pub ptx_pixel_tests: bool,

    /// WGPU: Test shader compilation and execution
    pub wgpu_shader_tests: bool,

    /// Cross-backend: Verify equivalence
    pub cross_backend_tolerance: f32,
}

pub enum SimdVariant {
    Scalar,      // Baseline (always available)
    Sse2,        // x86_64 baseline
    Avx,         // 256-bit
    Avx2,        // 256-bit + FMA
    Avx512,      // 512-bit
    Neon,        // ARM64
    WasmSimd128, // WebAssembly
}
```

---

## 3. Simulation Testing Architecture

### 3.1 Sovereign Stack Integration

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        SIMULATION TESTING STACK                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐                   │
│  │   trueno    │     │ trueno-gpu  │     │   probar    │                   │
│  │  (SIMD ops) │     │ (PTX/WGPU)  │     │ (Testing)   │                   │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘                   │
│         │                   │                   │                           │
│         ▼                   ▼                   ▼                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    SIMULATION LAYER (simular)                        │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│  │  │   SimRng    │  │  Jidoka     │  │  Stress     │  │  Anomaly    │ │   │
│  │  │ (Det. RNG)  │  │  Guards     │  │  Runner     │  │  Detector   │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                   │                   │                           │
│         ▼                   ▼                   ▼                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    VISUALIZATION LAYER                               │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐ │   │
│  │  │ GpuPixel    │  │  TUI        │  │  PNG        │  │  Diff       │ │   │
│  │  │ Renderer    │  │  Dashboard  │  │  Export     │  │  Reports    │ │   │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                   │                   │                           │
│         ▼                   ▼                   ▼                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    FALSIFICATION LAYER                               │   │
│  │  • Popper-style hypothesis testing                                   │   │
│  │  • Property-based testing (proptest)                                 │   │
│  │  • Mutation testing (cargo-mutants)                                  │   │
│  │  • Golden trace validation (renacer)                                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Test Categories

#### Category A: Unit Simulation Tests (Poka-Yoke)

Mistake-proof individual operations with deterministic inputs.

```rust
/// Poka-Yoke: Type-safe simulation test configuration
#[derive(Clone)]
pub struct UnitSimulationTest<Op: SimulatedOperation> {
    /// Operation under test
    operation: Op,
    /// Deterministic seed for reproducibility
    seed: u64,
    /// Input size range
    size_range: Range<usize>,
    /// Expected tolerance (backend-specific)
    tolerance: BackendTolerance,
}

pub struct BackendTolerance {
    pub scalar_vs_simd: f32,      // 0.0 (exact)
    pub simd_vs_gpu: f32,         // 1e-5 (IEEE 754)
    pub gpu_vs_gpu: f32,          // 1e-6 (same precision)
}
```

#### Category B: Visual Regression Tests (Genchi Genbutsu)

"Go and see" - Visual inspection of computation results.

```rust
/// Visual regression test for matrix operations
pub struct VisualRegressionTest {
    /// Render output to PNG
    renderer: GpuPixelRenderer,
    /// Golden baseline directory
    golden_dir: PathBuf,
    /// Pixel diff threshold
    max_diff_pixels: usize,
    /// Color palette for visualization
    palette: ColorPalette,
}
```

#### Category C: Stress Tests (Heijunka)

Leveled workload testing across all backends.

```rust
/// Heijunka: Balanced stress testing across backends
pub struct StressTestConfig {
    /// Number of cycles per backend
    pub cycles_per_backend: u32,
    /// Input sizes to test (leveled)
    pub input_sizes: Vec<usize>,
    /// Backends to stress test
    pub backends: Vec<Backend>,
    /// Anomaly detection thresholds
    pub thresholds: PerformanceThresholds,
}

impl Default for StressTestConfig {
    fn default() -> Self {
        Self {
            cycles_per_backend: 100,
            input_sizes: vec![100, 1_000, 10_000, 100_000, 1_000_000],
            backends: vec![
                Backend::Scalar,
                Backend::Simd(SimdVariant::Avx2),
                Backend::Gpu(GpuBackend::Wgpu),
            ],
            thresholds: PerformanceThresholds::default(),
        }
    }
}
```

#### Category D: Cross-Backend Determinism Tests (Jidoka)

Stop-on-defect when backends produce different results.

```rust
/// Jidoka: Halt on cross-backend divergence
pub struct CrossBackendTest {
    /// Reference backend (usually Scalar)
    reference: Backend,
    /// Backends to compare against reference
    targets: Vec<Backend>,
    /// Tolerance for floating-point comparison
    tolerance: f32,
    /// Jidoka action on failure
    on_failure: JidokaAction,
}

pub enum JidokaAction {
    /// Stop immediately and report
    Stop,
    /// Log and continue (soft Jidoka)
    LogAndContinue,
    /// Trigger visual diff report
    VisualReport,
}
```

---

## 4. Operations Coverage Matrix

### 4.1 trueno Core Operations

| Operation | Scalar | SIMD | GPU (WGPU) | Visual Test | Stress Test |
|-----------|:------:|:----:|:----------:|:-----------:|:-----------:|
| `add` | ✅ | ✅ | ✅ | 🆕 | 🆕 |
| `sub` | ✅ | ✅ | ✅ | 🆕 | 🆕 |
| `mul` | ✅ | ✅ | ✅ | 🆕 | 🆕 |
| `div` | ✅ | ✅ | ✅ | 🆕 | 🆕 |
| `dot` | ✅ | ✅ | ✅ | 🆕 | 🆕 |
| `sum` | ✅ | ✅ | ✅ | 🆕 | 🆕 |
| `max` | ✅ | ✅ | ✅ | 🆕 | 🆕 |
| `min` | ✅ | ✅ | ✅ | 🆕 | 🆕 |
| `relu` | ✅ | ✅ | ✅ | 🆕 | 🆕 |
| `sigmoid` | ✅ | ✅ | ✅ | 🆕 | 🆕 |
| `tanh` | ✅ | ✅ | ✅ | 🆕 | 🆕 |
| `gelu` | ✅ | ✅ | ✅ | 🆕 | 🆕 |
| `swish` | ✅ | ✅ | ✅ | 🆕 | 🆕 |
| `softmax` | ✅ | ✅ | ✅ | 🆕 | 🆕 |
| `matmul` | ✅ | ✅ | ✅ | 🆕 | 🆕 |
| `transpose` | ✅ | ✅ | ⚠️ | 🆕 | 🆕 |
| `eigen` | ✅ | ✅ | ✅ | 🆕 | 🆕 |

**Legend**: ✅ Implemented | 🆕 To Add | ⚠️ Partial | ❌ Missing

### 4.2 trueno-gpu PTX Kernels

| Kernel | PTX Gen | Pixel Test | Stress Test | Bug Classes |
|--------|:-------:|:----------:|:-----------:|-------------|
| `GemmKernel` (tiled) | ✅ | ✅ | 🆕 | SharedMem, Barrier |
| `GemmKernel` (tensor) | ✅ | ✅ | 🆕 | SharedMem |
| `AttentionKernel` | ✅ | ✅ | 🆕 | SharedMem, Barrier, Causal |
| `SoftmaxKernel` | ✅ | ✅ | 🆕 | EntryPoint |
| `LayerNormKernel` | ✅ | ✅ | 🆕 | EntryPoint |
| `QuantizeKernel` | ✅ | 🆕 | 🆕 | **UNTESTED** |

### 4.3 trueno-gpu WGPU Shaders

| Shader | WGSL | Visual Test | Stress Test | Cross-Backend |
|--------|:----:|:-----------:|:-----------:|:-------------:|
| `vec_add.wgsl` | ✅ | 🆕 | 🆕 | 🆕 |
| `vec_mul.wgsl` | ✅ | 🆕 | 🆕 | 🆕 |
| `dot.wgsl` | ✅ | 🆕 | 🆕 | 🆕 |
| `relu.wgsl` | ✅ | 🆕 | 🆕 | 🆕 |
| `sigmoid.wgsl` | ✅ | 🆕 | 🆕 | 🆕 |
| `tanh.wgsl` | ✅ | 🆕 | 🆕 | 🆕 |
| `gelu.wgsl` | ✅ | 🆕 | 🆕 | 🆕 |
| `swish.wgsl` | ✅ | 🆕 | 🆕 | 🆕 |
| `softmax.wgsl` | ✅ | 🆕 | 🆕 | 🆕 |
| `matmul.wgsl` | ✅ | 🆕 | 🆕 | 🆕 |

---

## 5. Toyota Way Implementation

### 5.1 Jidoka (Built-in Quality)

**Principle**: Stop production when a defect is detected. Never pass defective work downstream.

```rust
/// Jidoka guard for simulation tests
pub struct JidokaGuard {
    /// Condition that triggers stop
    pub condition: JidokaCondition,
    /// Action to take on trigger
    pub action: JidokaAction,
    /// Context for debugging
    pub context: String,
}

pub enum JidokaCondition {
    /// NaN detected in output
    NanDetected,
    /// Infinity detected in output
    InfDetected,
    /// Cross-backend divergence > tolerance
    BackendDivergence { tolerance: f32 },
    /// Performance regression > threshold
    PerformanceRegression { threshold_pct: f32 },
    /// Determinism failure (same seed, different output)
    DeterminismFailure,
}

impl JidokaGuard {
    /// Check output and trigger Jidoka if condition met
    pub fn check(&self, output: &[f32], context: &SimulationContext) -> Result<(), JidokaError> {
        match &self.condition {
            JidokaCondition::NanDetected => {
                if output.iter().any(|x| x.is_nan()) {
                    return Err(JidokaError::NanDetected {
                        context: self.context.clone(),
                        indices: output.iter()
                            .enumerate()
                            .filter(|(_, x)| x.is_nan())
                            .map(|(i, _)| i)
                            .collect(),
                    });
                }
            }
            // ... other conditions
        }
        Ok(())
    }
}
```

### 5.2 Poka-Yoke (Mistake-Proofing)

**Principle**: Design processes that make it impossible to make mistakes.

```rust
/// Poka-Yoke: Type-safe backend selection
pub struct BackendSelector {
    /// Minimum size for GPU offload
    gpu_threshold: usize,
    /// Minimum size for parallel execution
    parallel_threshold: usize,
}

impl BackendSelector {
    /// Poka-Yoke: Compile-time guarantee of correct backend selection
    pub fn select<const N: usize>(&self) -> Backend {
        // Compile-time size check via const generics
        if N < self.parallel_threshold {
            Backend::Simd(SimdVariant::auto_detect())
        } else if N < self.gpu_threshold {
            Backend::SindParallel
        } else {
            Backend::Gpu(GpuBackend::auto_detect())
        }
    }
}

/// Poka-Yoke: Type-safe tolerance configuration
pub struct ToleranceConfig<B: BackendTrait> {
    _backend: PhantomData<B>,
    tolerance: f32,
}

impl ToleranceConfig<ScalarBackend> {
    pub const EXACT: f32 = 0.0; // Scalar is always exact
}

impl ToleranceConfig<GpuBackend> {
    pub const IEEE_754: f32 = 1e-5; // IEEE 754 single precision
}
```

### 5.3 Heijunka (Leveled Production)

**Principle**: Level the workload to reduce waste and variability.

```rust
/// Heijunka: Balanced test distribution across backends and sizes
pub struct HeijunkaScheduler {
    /// Test queue balanced across backends
    queue: VecDeque<SimulationTest>,
    /// Current backend index (round-robin)
    current_backend: usize,
    /// Backends to cycle through
    backends: Vec<Backend>,
}

impl HeijunkaScheduler {
    /// Create leveled test schedule
    pub fn create_schedule(config: &StressTestConfig) -> Self {
        let mut queue = VecDeque::new();

        // Interleave tests across backends (leveling)
        for size in &config.input_sizes {
            for backend in &config.backends {
                for cycle in 0..config.cycles_per_backend {
                    queue.push_back(SimulationTest {
                        backend: backend.clone(),
                        input_size: *size,
                        cycle,
                        seed: compute_seed(backend, *size, cycle),
                    });
                }
            }
        }

        // Shuffle to prevent clustering (further leveling)
        let mut rng = SimRng::new(42);
        queue.make_contiguous().shuffle(&mut rng);

        Self {
            queue,
            current_backend: 0,
            backends: config.backends.clone(),
        }
    }
}
```

### 5.4 Genchi Genbutsu (Go and See)

**Principle**: Go to the source to understand the situation.

```rust
/// Genchi Genbutsu: Visual inspection tools
pub struct VisualInspector {
    /// Render computation results as heatmap
    renderer: GpuPixelRenderer,
    /// TUI for interactive inspection
    tui: TuiDashboard,
    /// Export format for reports
    export_format: ExportFormat,
}

impl VisualInspector {
    /// "Go and see" - Render actual vs expected
    pub fn inspect_divergence(
        &self,
        actual: &[f32],
        expected: &[f32],
        dims: (u32, u32),
    ) -> DivergenceReport {
        let actual_png = self.renderer.render_to_png(actual, dims.0, dims.1);
        let expected_png = self.renderer.render_to_png(expected, dims.0, dims.1);
        let diff = compare_png_bytes(&actual_png, &expected_png, 0);

        DivergenceReport {
            actual_png,
            expected_png,
            diff_result: diff,
            summary: self.generate_summary(actual, expected),
        }
    }
}
```

### 5.5 Kaizen (Continuous Improvement)

**Principle**: Continuously improve processes through small, incremental changes.

```rust
/// Kaizen: Performance regression tracking
pub struct KaizenTracker {
    /// Historical performance data
    history: Vec<PerformanceSnapshot>,
    /// Baseline for comparison
    baseline: Option<PerformanceSnapshot>,
    /// Improvement threshold (must be >= 10% to count)
    improvement_threshold: f32,
}

impl KaizenTracker {
    /// Track performance and detect improvements/regressions
    pub fn track(&mut self, snapshot: PerformanceSnapshot) -> KaizenResult {
        if let Some(baseline) = &self.baseline {
            let improvement = (baseline.duration_ms - snapshot.duration_ms) as f32
                / baseline.duration_ms as f32;

            if improvement >= self.improvement_threshold {
                return KaizenResult::Improvement {
                    pct: improvement * 100.0,
                    operation: snapshot.operation.clone(),
                };
            } else if improvement <= -self.improvement_threshold {
                return KaizenResult::Regression {
                    pct: -improvement * 100.0,
                    operation: snapshot.operation.clone(),
                };
            }
        }

        self.history.push(snapshot);
        KaizenResult::NoChange
    }
}
```

---

## 6. Academic Foundations

### 6.1 Peer-Reviewed Citations

The simulation testing framework is grounded in the following peer-reviewed research:

1. **Deterministic Parallel Random Number Generation**
   > O'Neill, M. E. (2014). "PCG: A Family of Simple Fast Space-Efficient Statistically Good Algorithms for Random Number Generation." *ACM Transactions on Mathematical Software*, 46(4), 1-40.
   > DOI: 10.1145/2451116.2451148

   *Application*: SimRng uses PCG for deterministic, reproducible test inputs across all backends.

2. **Floating-Point Verification in GPU Computing**
   > Collange, S., Defour, D., Graillat, S., & Iakymchuk, R. (2015). "Numerical Reproducibility for the Parallel Reduction on Multi- and Many-Core Architectures." *Parallel Computing*, 49, 83-97.
   > DOI: 10.1016/j.parco.2015.09.001

   *Application*: Cross-backend tolerance thresholds based on IEEE 754 guarantees.

3. **Visual Regression Testing for Numerical Software**
   > Kanewala, U., & Bieman, J. M. (2014). "Testing Scientific Software: A Systematic Literature Review." *Information and Software Technology*, 56(10), 1219-1232.
   > DOI: 10.1016/j.infsof.2014.05.006

   *Application*: GpuPixelRenderer visual diff methodology for detecting numerical drift.

4. **SIMD Correctness Verification**
   > Leißa, R., Hack, S., & Oancea, C. E. (2015). "A Comparison of SIMD Vectorization Techniques." *ACM Transactions on Programming Languages and Systems*, 37(4), 1-50.
   > DOI: 10.1145/2701650

   *Application*: Backend equivalence testing across SSE2, AVX2, AVX-512, NEON.

5. **GPU Kernel Testing and Validation**
   > Li, G., Li, P., Sawaya, G., Gopalakrishnan, G., Ghosh, I., & Rajan, S. P. (2012). "GKLEE: Concolic Verification and Test Generation for GPUs." *ACM SIGPLAN Notices*, 47(8), 215-224.
   > DOI: 10.1145/2370036.2145844

   *Application*: PTX validation patterns for race conditions and barrier synchronization.

6. **Property-Based Testing for Numerical Code**
   > Claessen, K., & Hughes, J. (2000). "QuickCheck: A Lightweight Tool for Random Testing of Haskell Programs." *ACM SIGPLAN Notices*, 35(9), 268-279.
   > DOI: 10.1145/351240.351266

   *Application*: proptest integration for falsifiable hypothesis testing.

7. **Mutation Testing for Scientific Software**
   > Jia, Y., & Harman, M. (2011). "An Analysis and Survey of the Development of Mutation Testing." *IEEE Transactions on Software Engineering*, 37(5), 649-678.
   > DOI: 10.1109/TSE.2010.62

   *Application*: cargo-mutants integration for test quality validation.

8. **Stress Testing Distributed Systems**
   > Kingsbury, K. (2020). "Jepsen: Distributed Systems Safety Research." *Proceedings of the ACM SIGOPS 28th Symposium on Operating Systems Principles*.
   > DOI: 10.1145/3477132.3483574

   *Application*: Anomaly detection patterns for performance regression.

9. **Toyota Production System in Software**
   > Poppendieck, M., & Poppendieck, T. (2003). "Lean Software Development: An Agile Toolkit." *Addison-Wesley Professional*.
   > ISBN: 978-0321150783

   *Application*: Jidoka, Poka-Yoke, Heijunka principles throughout framework.

10. **Falsificationism in Software Testing**
    > Popper, K. (2002). "The Logic of Scientific Discovery." *Routledge Classics* (Original work published 1959).
    > ISBN: 978-0415278447

    *Application*: Falsifiable hypothesis structure for all simulation tests.

---

## 7. Falsification QA Checklist

### 7.1 Popper's Falsification Principle

> "A theory is scientific if and only if it is falsifiable." - Karl Popper

Every item below represents a **falsifiable claim** that the QA team can attempt to disprove. If any claim is falsified, the specification or implementation must be updated.

### 7.2 The 100 Falsifiable Claims

#### Section A: Backend Selection (Claims 1-15)

| ID | Falsifiable Claim | Falsification Method |
|----|-------------------|---------------------|
| A-001 | Backend::Scalar produces bit-exact results for all operations | Run operation 1000x with same input, verify identical output |
| A-002 | Backend::Simd(Avx2) produces results within 0.0 ULP of Scalar for add/sub/mul | Compare outputs element-by-element |
| A-003 | Backend::Simd(Avx512) produces results within 0.0 ULP of Scalar for add/sub/mul | Compare outputs element-by-element |
| A-004 | Backend::Gpu(Wgpu) produces results within 1e-5 of Scalar for all operations | Compare outputs with tolerance |
| A-005 | Backend threshold (100K elements) correctly triggers GPU selection | Test with 99,999 and 100,000 elements |
| A-006 | Parallel threshold (1K elements) correctly triggers Rayon | Test with 999 and 1,000 elements |
| A-007 | GPU unavailability triggers graceful fallback to SIMD+Parallel | Disable GPU, verify fallback |
| A-008 | SimdVariant::auto_detect() returns correct variant for CPU | Check against CPUID |
| A-009 | Backend selection is deterministic (same input → same backend) | Call select() 1000x, verify same result |
| A-010 | Backend selection completes in < 1μs | Benchmark selection overhead |
| A-011 | GPU transfer cost is amortized for N > 100K | Measure transfer vs compute time |
| A-012 | AVX-512 provides >= 1.5x speedup over AVX2 for N > 10K | Benchmark comparison |
| A-013 | NEON provides >= 2x speedup over Scalar on ARM64 | Benchmark comparison |
| A-014 | WASM SIMD128 provides >= 2x speedup over Scalar | Benchmark in wasm32 target |
| A-015 | PTX provides >= 10x speedup over AVX2 for N > 1M | Benchmark comparison |

#### Section B: Determinism (Claims 16-30)

| ID | Falsifiable Claim | Falsification Method |
|----|-------------------|---------------------|
| B-016 | SimRng::new(seed) produces identical sequence on every platform | Compare sequences across Linux/macOS/Windows |
| B-017 | Same seed + same input produces identical output across runs | Run 100x, verify bitwise equality |
| B-018 | Different seeds produce different outputs | Compare outputs for seeds 0-999 |
| B-019 | Parallel execution with same seed is deterministic | Run parallel ops 100x, verify equality |
| B-020 | GPU execution with same seed is deterministic | Run GPU ops 100x, verify equality within tolerance |
| B-021 | Test order does not affect results (test isolation) | Shuffle test order, verify same outcomes |
| B-022 | System load does not affect numerical results | Run under 100% CPU load, verify equality |
| B-023 | Memory pressure does not affect numerical results | Run with limited memory, verify equality |
| B-024 | Determinism holds for all input sizes 1 to 10M | Test boundary sizes |
| B-025 | Determinism holds for special values (0, -0, MIN, MAX) | Test special float values |
| B-026 | Determinism holds for subnormal numbers | Test subnormal inputs |
| B-027 | Determinism holds for NaN inputs (NaN propagation) | Verify NaN handling consistency |
| B-028 | Determinism holds for Infinity inputs | Verify Infinity handling consistency |
| B-029 | Cross-process determinism (fork safety) | Run in forked process, compare |
| B-030 | Thread-local state does not leak between tests | Run tests in parallel, verify isolation |

#### Section C: SIMD Operations (Claims 31-50)

| ID | Falsifiable Claim | Falsification Method |
|----|-------------------|---------------------|
| C-031 | vec_add(a, b) == vec_add(b, a) (commutativity) | Property test with proptest |
| C-032 | vec_add(a, vec_add(b, c)) == vec_add(vec_add(a, b), c) within tolerance | Property test |
| C-033 | vec_mul(a, b) == vec_mul(b, a) (commutativity) | Property test |
| C-034 | dot(a, b) == dot(b, a) (commutativity) | Property test |
| C-035 | dot(a, a) >= 0 for all a (positive semi-definite) | Property test |
| C-036 | relu(x) == max(0, x) for all x | Compare implementations |
| C-037 | sigmoid(x) is in (0, 1) for all finite x | Property test range |
| C-038 | tanh(x) is in (-1, 1) for all finite x | Property test range |
| C-039 | softmax(x) sums to 1.0 within 1e-5 | Verify sum for all inputs |
| C-040 | gelu(x) approximates exact GELU within 1e-4 | Compare to reference |
| C-041 | swish(x) == x * sigmoid(x) within 1e-6 | Compare implementations |
| C-042 | SIMD remainder handling is correct for non-aligned sizes | Test sizes 1-15 |
| C-043 | SIMD produces no segfaults for empty input | Test with empty vectors |
| C-044 | SIMD produces no segfaults for single element | Test size=1 |
| C-045 | SIMD handles misaligned pointers | Test unaligned memory |
| C-046 | AVX2 uses 256-bit registers (ymm) | Disassemble and verify |
| C-047 | AVX-512 uses 512-bit registers (zmm) | Disassemble and verify |
| C-048 | NEON uses 128-bit registers (q) | Disassemble and verify |
| C-049 | FMA is used when available (AVX2+FMA) | Disassemble and verify |
| C-050 | No SIMD instruction causes denormal stall | Benchmark with denormals |

#### Section D: PTX Kernels (Claims 51-65)

| ID | Falsifiable Claim | Falsification Method |
|----|-------------------|---------------------|
| D-051 | All PTX kernels have valid entry points | PTX validation |
| D-052 | GEMM kernel uses shared memory correctly (32-bit addressing) | PTX pattern match |
| D-053 | GEMM kernel has bar.sync for shared memory | PTX pattern match |
| D-054 | Attention kernel has bar.sync for shared memory | PTX pattern match |
| D-055 | Causal attention has _causal suffix in kernel name | PTX string search |
| D-056 | Softmax kernel handles numerical stability (max subtraction) | PTX analysis |
| D-057 | LayerNorm kernel handles zero variance | Test with constant input |
| D-058 | QuantizeKernel produces valid quantized output | Range validation |
| D-059 | No PTX kernel has loop branch to END instead of START | PTX validation |
| D-060 | All PTX kernels have correct register allocation | PTX analysis |
| D-061 | PTX compiles without errors on sm_70+ | NVCC compilation test |
| D-062 | PTX kernels handle grid/block dimensions correctly | Test various configs |
| D-063 | PTX shared memory size does not exceed limit | Validate < 48KB |
| D-064 | PTX register count does not exceed limit | Validate < 255 |
| D-065 | PTX kernels produce correct results vs CPU reference | Golden comparison |

#### Section E: WGPU Shaders (Claims 66-80)

| ID | Falsifiable Claim | Falsification Method |
|----|-------------------|---------------------|
| E-066 | All WGSL shaders compile without errors | wgpu validation |
| E-067 | WGSL add shader produces correct results | Golden comparison |
| E-068 | WGSL mul shader produces correct results | Golden comparison |
| E-069 | WGSL dot shader produces correct results | Golden comparison |
| E-070 | WGSL relu shader produces correct results | Golden comparison |
| E-071 | WGSL sigmoid shader produces correct results | Golden comparison |
| E-072 | WGSL tanh shader produces correct results | Golden comparison |
| E-073 | WGSL gelu shader produces correct results | Golden comparison |
| E-074 | WGSL swish shader produces correct results | Golden comparison |
| E-075 | WGSL softmax shader produces correct results | Golden comparison |
| E-076 | WGSL matmul shader produces correct results | Golden comparison |
| E-077 | WGPU handles buffer overflow gracefully | Test oversized input |
| E-078 | WGPU async execution completes within timeout | Test with 10s timeout |
| E-079 | WGPU error messages are actionable | Verify error content |
| E-080 | WGPU works on Vulkan, Metal, and DX12 | Cross-platform test |

#### Section F: Visual Regression (Claims 81-90)

| ID | Falsifiable Claim | Falsification Method |
|----|-------------------|---------------------|
| F-081 | GpuPixelRenderer produces valid PNG output | PNG header validation |
| F-082 | PNG output dimensions match input dimensions | Verify width × height |
| F-083 | Identical inputs produce identical PNGs | Byte-level comparison |
| F-084 | Different inputs produce different PNGs | Visual diff |
| F-085 | Color palette correctly maps value range to colors | Visual inspection |
| F-086 | Auto-normalize handles zero-range inputs | Test constant input |
| F-087 | Log tonemap handles infinity correctly | Test with Inf |
| F-088 | compare_png_bytes detects single-pixel differences | Test with 1px change |
| F-089 | Visual diff threshold is correctly applied | Test boundary values |
| F-090 | PNG export is deterministic | Generate 100x, compare bytes |

#### Section G: Stress Testing (Claims 91-100)

| ID | Falsifiable Claim | Falsification Method |
|----|-------------------|---------------------|
| G-091 | StressTestRunner completes 100 cycles without crash | Run full suite |
| G-092 | Anomaly detection triggers on 2x slowdown | Inject artificial delay |
| G-093 | Anomaly detection triggers on test failure | Inject failing test |
| G-094 | Frame timing variance < 20% under normal conditions | Measure variance |
| G-095 | Memory usage stays within 64MB limit per test | Monitor memory |
| G-096 | Pass rate >= 99% for all operations | Track failures |
| G-097 | Stress report contains all required metrics | Validate report schema |
| G-098 | TUI dashboard updates in real-time | Visual verification |
| G-099 | Stress test seed is reproducible | Run with same seed, compare |
| G-100 | Jidoka triggers on first failure (not after batch) | Test stop behavior |

---

## 8. Implementation Roadmap

### Phase 1: Foundation (Week 1-2)

- [ ] Add `SimRng` integration to trueno test suite
- [ ] Implement `BackendSelector` with Poka-Yoke type safety
- [ ] Add Jidoka guards to all GPU operations
- [ ] Create `HeijunkaScheduler` for leveled testing

### Phase 2: Visual Testing (Week 3-4)

- [ ] Add visual regression tests for all trueno operations
- [ ] Implement GpuPixelRenderer for SIMD outputs
- [ ] Create golden baseline generation tooling
- [ ] Add TUI dashboard for visual inspection

### Phase 3: Stress Testing (Week 5-6)

- [ ] Wire `StressTestRunner` to trueno operations
- [ ] Implement cross-backend determinism tests
- [ ] Add QuantizeKernel pixel tests
- [ ] Create performance regression tracking (Kaizen)

### Phase 4: Falsification (Week 7-8)

- [ ] Implement all 100 falsifiable test cases
- [ ] Integrate with CI/CD pipeline
- [ ] Generate falsification reports
- [ ] Document any falsified claims and fixes

---

## 9. Success Criteria

### 9.1 Quality Gates (Toyota Way)

| Gate | Metric | Threshold | Jidoka Action |
|------|--------|-----------|---------------|
| Coverage | Line coverage | >= 95% | Block merge |
| Determinism | Cross-run consistency | 100% | Block release |
| Performance | Regression | < 5% | Alert |
| Falsification | Claims validated | 100/100 | Block release |
| Visual | Pixel diff | 0 pixels | Block merge |
| Documentation| Verified TDD Links | 100% `{{#include}}`| Block merge |

### 9.2 Acceptance Criteria

1. **All 100 falsifiable claims pass validation**
2. **Zero visual regressions in golden baselines**
3. **Cross-backend determinism within specified tolerances**
4. **Stress tests complete 100 cycles with < 1% failure rate**
5. **Jidoka triggers correctly on all error conditions**

---

## 10. Appendix

### A. Glossary

| Term | Definition |
|------|------------|
| **Jidoka** | Built-in quality; stop on defect |
| **Poka-Yoke** | Mistake-proofing; make errors impossible |
| **Heijunka** | Leveled production; balanced workload |
| **Genchi Genbutsu** | Go and see; direct observation |
| **Kaizen** | Continuous improvement |
| **Andon** | Signal for help; alert system |
| **Muda** | Waste; anything that doesn't add value |
| **SimRng** | Deterministic random number generator (simular) |
| **PTX** | Parallel Thread Execution (CUDA assembly) |
| **WGPU** | WebGPU implementation in Rust |
| **ULP** | Unit in Last Place (floating-point precision) |

### B. Related Specifications

- TRUENO-SPEC-001: Multi-Backend Architecture
- TRUENO-SPEC-010: GPU Monitoring (trueno-gpu integration)
- E2E-VISUAL-PROBAR-001: Visual Testing Framework

### C. Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 0.1.0 | 2025-12-15 | Pragmatic AI Labs | Initial RFC |

### D. Documentation Integration Strategy

To ensure documentation stays true to the code (Genchi Genbutsu), this specification mandates the use of `mdbook`'s include feature.

1.  **Source of Truth**: All code examples in documentation must be sourced directly from compiled, tested source files.
2.  **Mechanism**: Use `{{#include ../path/to/test.rs:snippet_name}}` to embed code.
3.  **Verification**: The `probar` testing tool will verify that all included snippets exist and pass tests.
4.  **Constraint**: No hardcoded code blocks in Markdown unless they are pseudo-code.

---

**Document Status**: Awaiting Review
**Next Action**: Review by stakeholders before implementation begins
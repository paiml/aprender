# Trueno Roadmap (PMAT-Driven)

**Strategic Vision**: PyTorch/NumPy replacement for Rust with EXTREME TDD quality gates

**📖 Comprehensive Spec**: [PyTorch/NumPy Replacement Specification](docs/specifications/pytorch-numpy-replacement-spec.md)

---

## Current State: v0.2.2 (2025-11-18)

### Position Analysis

**NumPy Replacement**: ~35% Complete
- ✅ What Works: 1D ops, reductions, SIMD/GPU acceleration
- ❌ Critical Gaps: Multi-dim arrays, broadcasting, advanced indexing

**PyTorch Replacement**: ~15% Complete
- ✅ What Works: GPU activations (14 ops), inference only
- ❌ Critical Blockers: No autograd, no layers, no training capability

### Core Capabilities (v0.2.0)

```
✅ 1D Vector<f32> type
✅ CPU SIMD backends (SSE2/AVX/AVX2/NEON)
✅ GPU backend (wgpu: Vulkan/Metal/DX12/WebGPU)
✅ 14 GPU-accelerated operations
✅ Runtime dispatch (auto-select best backend)
✅ EXTREME TDD (>90% coverage, mutation testing)
```

**GPU Operations by Complexity**:
- **Low** (>100K threshold): vec_add, dot, relu, leaky_relu, elu, sigmoid, tanh, swish, GELU, clip
- **Medium** (>10K threshold): softmax, log_softmax
- **High** (>1K threshold): matmul, convolve2d

### Quality Metrics (Current)

```
Test Coverage:     >90%
Mutation Testing:  80%+ kill rate
PMAT TDG Grade:    A (92.1/100)
Repo Score:        90/110
GPU Speedup:       ⚠️ Matmul ONLY 2-10x (13/14 ops slower, see analysis)
Total Tests:       889 tests (759 unit + 21 integration + 109 doc)
```

---

## Phase 1: Complete 1D Operations
**Timeline**: v0.2.x → v0.3.0 (2-3 months)
**Goal**: Best-in-class 1D vector compute
**Toyota Way**: *Jidoka* (完成 - Complete current work before starting new work)

### v0.2.1 (Next 2 Weeks) - CURRENT SPRINT

#### Deliverables

- [x] **GPU softmax/log_softmax** ✅ COMPLETE
  - 5 WGSL shaders (max/sum reduction, exp-subtract, normalize, log_softmax)
  - 4-pass multi-pass coordination (async/await)
  - 18 tests pass (unit + property-based)
  - Benchmarks: 10K, 100K, 1M sizes
  - README documentation with examples
  - Actual speedup: 2-20x over scalar

- [x] **Benchmark all GPU ops** ✅ COMPLETE - *Genchi Genbutsu* (現地現物 - Go see for yourself)
  - Measured 40+ configurations across 14 operations (1K-1M elements)
  - **CRITICAL FINDING**: GPU UNSUITABLE for 13/14 operations
  - ✅ Matmul: 2-10x speedup (500×500+)
  - ❌ All element-wise: 2-65,000x SLOWER (transfer overhead dominates)
  - Root cause: 14-55ms fixed GPU overhead >> compute time
  - Full analysis: [docs/performance-analysis.md](docs/performance-analysis.md)
  - **Decision**: Disable GPU for element-wise ops, focus on SIMD

- [x] **Performance regression suite** ✅ COMPLETE
  - Baseline saved: `.performance-baselines/baseline-current.txt`
  - Framework: `.performance-baselines/README.md`, `baseline-template.json`
  - Makefile targets: `bench-save-baseline`, `bench-compare`, `bench-gpu`
  - **Status**: Infrastructure ready, CI integration pending

- [x] **Implement GPU strategic decision** ✅ COMPLETE
  - Set GPU_THRESHOLD = usize::MAX for 10 activation functions
  - Lowered matmul threshold: 1000 → 500 (empirical data)
  - GPU now used ONLY for matmul ≥500×500 (2-10x speedup)
  - All element-wise ops use scalar/SIMD only
  - **Result**: Eliminated 2-65,000x slowdowns on activation functions

#### Quality Gates (v0.2.1)

```
Required for Release:
✅ All GPU ops benchmarked (validate claims)
✅ Performance regression suite in CI
✅ Test coverage ≥90%
✅ Mutation testing ≥80%
✅ Zero clippy warnings
✅ PMAT TDG ≥B+ (85/100)
✅ Repo score ≥90/110
```

---

### v0.2.2 - v0.2.5 (6-8 Weeks)

**Strategy Pivot**: Focus on SIMD optimization (GPU unsuitable for element-wise ops)

#### Deliverables

- [x] **Remaining activations** (SIMD-optimized, NO GPU) ✅ **COMPLETE**
  - ✅ hardswish (MobileNetV3) - commit 3130859
  - ✅ mish (modern swish alternative) - commit 482737d
  - ✅ selu (self-normalizing networks) - commit 94c12d0
  - **Result**: 33 tests (18 unit + 15 property), all passing
  - **Note**: GPU disabled per v0.2.1 analysis (was 800x slower)

- [x] **Scalar reductions implemented** ✅ **COMPLETE**
  - ✅ argmax/argmin - working scalar implementations
  - ✅ sum/mean/variance/stddev - working scalar implementations
  - **Next**: SIMD optimization (parallel reduction + index tracking)
  - **Success Criteria**: SIMD speedup ≥2-4x vs scalar (benchmark needed)

- [x] **Scalar unary ops implemented** ✅ **COMPLETE**
  - ✅ exp/ln/log2/log10/pow/sqrt - all working scalar implementations
  - **Next**: SIMD optimization (vectorized math functions)
  - **Success Criteria**: SIMD speedup ≥2-4x vs scalar (benchmark needed)
  - **Note**: GPU disabled (transfer overhead dominates)

- [x] **Performance regression CI** ✅ **COMPLETE**
  - ✅ Created `scripts/check_regression.py` (parses Criterion output)
  - ✅ Updated `make bench-compare` to use script
  - ✅ Integrated into CI workflow (`.github/workflows/ci.yml`)
  - **Success Criteria**: Detect >5% regressions automatically

- [x] **SIMD optimization: norm_linf** ✅ **COMPLETE** - *Kaizen* (改善 - Quick wins first)
  - ✅ Eliminated temporary vector allocation (13-43% scalar speedup)
  - ✅ Single-pass AVX2 abs+max (8-way parallel, bitwise AND + max)
  - ✅ Single-pass SSE2 abs+max (4-way parallel)
  - ✅ Horizontal reduction with 128-bit halves extraction
  - **Result**: 1.1-3.2x total speedup across all sizes
  - **Benchmarks**: 100 elem 3.2x, 1K 3.0x, 10K 2.1x, 100K 2.1x
  - **Next**: Continue SIMD optimization for other reduction ops

#### Quality Gates (v0.2.2-v0.2.5)

```
Required for Each Release:
✅ EXTREME TDD cycle for each operation:
  - Implementation → Tests → Benchmarks → Documentation
✅ Gradient checking (prepare for Phase 3 autograd)
✅ Backend equivalence: SIMD vs Scalar (< 1e-5 error)
✅ Test coverage ≥90%
✅ Mutation testing ≥80%
✅ No performance regressions >5%
```

---

### v0.3.0: 1D Operations Complete (Milestone)

**Target**: NumPy ~40%, PyTorch ~18%

#### Deliverables

- [ ] **Async GPU API** - *Kaizen* (改善 - Continuous improvement)
  - Batch multiple operations to reduce transfer overhead
  - Async execution with futures
  - **Success Criteria**: 2x fewer GPU transfers for chained ops

- [ ] **CPU backend optimizations**
  - AVX-512 support (Zen4/Sapphire Rapids+)
  - Better auto-vectorization hints
  - **Success Criteria**: 8x speedup over scalar (AVX-512)

- [x] **WASM SIMD128** ✅ **COMPLETE**
  - Browser deployment support
  - SIMD implementations for all VectorBackend operations:
    - Element-wise: add, sub, mul, div, abs, scale, clamp
    - Reductions: sum, max, min, argmax, argmin, dot, norm_l1, norm_l2, norm_linf
    - Activations: relu, exp, sigmoid, gelu, swish, tanh (with SIMD exp approximation)
    - Interpolation: lerp, fma
  - **Success Criteria**: 2x speedup over scalar (WASM) ✅ Achieved via SIMD128

- [ ] **Comprehensive benchmarks**
  - vs NumPy (for 1D ops)
  - vs PyTorch (for activations)
  - Publish results in README
  - **Success Criteria**: Within 20% of NumPy/PyTorch for 1D ops

#### Success Metrics (v0.3.0 Phase Gate)

```
Technical:
✅ All common 1D operations GPU-accelerated (20+ ops)
✅ 10-50x GPU speedup validated by benchmarks
✅ Async GPU API reduces transfer overhead by 2x
✅ AVX-512 backend: 8x speedup over scalar
✅ WASM SIMD128: 2x speedup over scalar

Quality:
✅ Test coverage ≥90%
✅ Mutation testing ≥80%
✅ PMAT TDG ≥A- (92/100)
✅ Repo score ≥95/110

Adoption:
✅ Used in production by ≥3 projects
✅ ≥100 GitHub stars
✅ ≥10 contributors
```

**🚨 Phase Gate Decision Point**: Proceed to Phase 2 only if ALL success metrics achieved

---

## Phase 2: Multi-Dimensional Tensors
**Timeline**: v0.4.0 → v0.6.0 (6-12 months)
**Goal**: NumPy-competitive for 2D/3D arrays
**Toyota Way**: *Heijunka* (平準化 - Level loading - balance implementation with validation)

### v0.4.0: Tensor Type Foundation (3-4 Months)

**Target**: NumPy ~50%, PyTorch ~20%

#### Deliverables

- [ ] **`Tensor<T, const N: usize>` type**
  - Const generics for rank (compile-time safety)
  - Row-major storage (C-contiguous, NumPy-compatible)
  - Strides-based layout (zero-copy transpose)
  - Views vs owned data (Arc-based sharing)
  - **Success Criteria**: Represent 0D-4D tensors with compile-time rank verification

- [ ] **2D operations**
  - Transpose (zero-copy via stride swap)
  - Reshape, flatten
  - Row/column slicing
  - Optimized 2D matmul (GPU-accelerated)
  - **Success Criteria**: 80-120% of NumPy speed for 2D ops

- [ ] **Storage design validation** - *Genchi Genbutsu*
  - Benchmark row-major vs column-major layouts
  - Validate zero-copy transpose performance
  - **Success Criteria**: Zero-copy transpose 100x faster than data reorganization

#### Quality Gates (v0.4.0)

```
Required:
✅ Differential testing: All ops vs NumPy (< 1e-5 error)
✅ Property-based tests: Shape transformations
✅ Backend equivalence: GPU vs CPU for 2D ops
✅ Test coverage ≥90%
✅ Mutation testing ≥80%
✅ PMAT TDG ≥A- (92/100)

Design Validation:
✅ Const generics enable compile-time shape checking
✅ Strides enable zero-copy operations
✅ Memory layout optimized for BLAS/GPU performance
```

---

### v0.5.0: Broadcasting (2-3 Months)

**Target**: NumPy ~65%, PyTorch ~20%

#### Deliverables

- [ ] **NumPy-compatible broadcasting**
  - Shape compatibility checking
  - Fused GPU kernels (avoid materializing intermediates)
  - Element-wise ops with broadcasting
  - **Success Criteria**: Pass 80%+ of NumPy broadcasting tests

- [ ] **Advanced indexing**
  - Boolean masking
  - Integer array indexing
  - Slicing syntax (`[1:5, ::2]` via macro)
  - **Success Criteria**: NumPy-style indexing ergonomics

#### Quality Gates (v0.5.0)

```
Required - Jidoka (Build Quality In):
✅ Property-based testing vs NumPy (differential testing)
✅ Fused broadcasting kernels (zero intermediate allocation)
✅ Test coverage ≥90%
✅ Mutation testing ≥80%

Broadcasting Validation:
✅ Matches NumPy broadcasting semantics exactly
✅ Fused kernels 2x faster than naive implementation
✅ No memory overhead for broadcasted operations
```

---

### v0.6.0: NumPy Parity (3-4 Months)

**Target**: NumPy ~80%, PyTorch ~20% (Milestone)

#### Deliverables

- [ ] **Generic dtype support**
  - f16, f32, f64, i32, i64, u32, etc.
  - Trait-based implementation
  - **Success Criteria**: Support 10+ data types

- [ ] **NumPy-style API**
  - Creation: zeros, ones, arange, linspace
  - Manipulation: concatenate, stack, split
  - Conditional: where, argwhere
  - **Success Criteria**: 80%+ API coverage for core operations

- [ ] **NumPy test suite validation** - *Genchi Genbutsu*
  - Run NumPy test suite against Trueno
  - **Success Criteria**: Pass 80%+ of NumPy tests (for covered ops)

#### Success Metrics (v0.6.0 Phase Gate)

```
Technical:
✅ 80-120% of NumPy performance (within 20%)
✅ Support 0D-4D tensors, 10+ data types
✅ Broadcasting with fused GPU kernels
✅ Pass 80%+ of NumPy test suite (covered ops)

Quality:
✅ Differential testing: All ops vs NumPy
✅ Test coverage ≥90%
✅ Mutation testing ≥80%
✅ PMAT TDG ≥A (94/100)
✅ Repo score ≥100/110

Adoption:
✅ ≥10 production deployments
✅ ≥500 GitHub stars
✅ ≥50 contributors
```

**🚨 Phase Gate Decision Point**: Proceed to Phase 3 only if ALL success metrics achieved

---

## Phase 3: Autograd & Training
**Timeline**: v0.7.0 → v1.0.0 (12-18 months)
**Goal**: PyTorch-competitive for training
**Toyota Way**: *Jidoka* (自働化 - Automation with human touch - halt on defects)

### v0.7.0: Autograd Engine (4-6 Months)

**Target**: NumPy ~80%, PyTorch ~35%

#### Deliverables

- [ ] **Reverse-mode AD engine**
  - Dynamic graph construction (PyTorch-style)
  - Gradient tape with backward functions
  - **Success Criteria**: Compute gradients for all operations

- [ ] **Gradient checking** - *Jidoka* (CRITICAL QUALITY GATE)
  - Automatic verification: analytical vs numerical gradients
  - Required for EVERY operation with autograd
  - **Success Criteria**: All gradients match numerical within 1e-4

- [ ] **Core ops with gradients**
  - All element-wise ops (add, mul, exp, log, etc.)
  - Reductions (sum, mean, max)
  - Linear algebra (matmul, conv2d)
  - All 14+ activations
  - **Success Criteria**: Gradients match PyTorch (< 1e-5 error)

- [ ] **Memory optimization**
  - Gradient checkpointing
  - In-place operations where safe
  - **Success Criteria**: Train 50-layer network without OOM

#### Quality Gates (v0.7.0)

```
Required - HALT THE LINE ON GRADIENT BUGS:
✅ Gradient checking: EVERY operation (automated)
✅ Differential testing: Gradients vs PyTorch (< 1e-5 error)
✅ Property-based tests: Chain rule, linearity
✅ Fuzz testing: Gradient computation robustness
✅ Test coverage ≥90%
✅ Mutation testing ≥80%

Autograd Validation:
✅ No silent gradient failures
✅ Backward pass matches PyTorch exactly
✅ Memory-efficient (gradient checkpointing works)
```

---

### v0.8.0: Neural Network Layers (3-4 Months)

**Target**: NumPy ~80%, PyTorch ~50%

#### Deliverables

- [ ] **nn::Module trait**
  - Parameter tracking
  - Forward/backward hooks
  - **Success Criteria**: Ergonomic layer composition

- [ ] **Core layers**
  - Linear, Conv2d, MaxPool2d
  - BatchNorm, LayerNorm
  - Dropout
  - **Success Criteria**: Match PyTorch API ergonomics

- [ ] **Loss functions**
  - CrossEntropyLoss, MSELoss, BCELoss
  - **Success Criteria**: Numerical match with PyTorch

#### Quality Gates (v0.8.0)

```
Required:
✅ Differential testing: All layers vs PyTorch
✅ Gradient checking: All layers
✅ Can build ResNet-18, BERT-base
✅ Test coverage ≥90%
✅ Mutation testing ≥80%
```

---

### v0.9.0: Optimizers (2-3 Months)

**Target**: NumPy ~80%, PyTorch ~55%

#### Deliverables

- [ ] **Core optimizers**
  - SGD (momentum, Nesterov)
  - Adam (weight decay, AMSGrad)
  - AdamW, RMSprop
  - **Success Criteria**: Match PyTorch update rules exactly

- [ ] **Learning rate schedulers**
  - StepLR, ExponentialLR, CosineAnnealing
  - **Success Criteria**: Match PyTorch scheduling exactly

#### Quality Gates (v0.9.0)

```
Required:
✅ Differential testing: Optimizer updates vs PyTorch
✅ Can train ResNet-50 to convergence
✅ Learning curves match PyTorch
✅ Test coverage ≥90%
```

---

### v1.0.0: Training-Ready (3-4 Months) - MAJOR MILESTONE

**Target**: NumPy ~80%, PyTorch ~60%

#### Deliverables

- [ ] **Model serialization**
  - Save/load checkpoints (state_dict)
  - ONNX export
  - **Success Criteria**: Load PyTorch weights, export to ONNX

- [ ] **Distributed training**
  - Data parallelism
  - Gradient synchronization (AllReduce)
  - **Success Criteria**: Linear scaling to 4 GPUs

- [ ] **Production features**
  - Mixed precision (FP16/BF16)
  - Gradient clipping
  - Early stopping
  - **Success Criteria**: Train production models end-to-end

- [ ] **Model hub** - Combat ecosystem lock-in
  - ResNet-{18,34,50}, BERT-base, MobileNetV2
  - Pretrained weights (converted from PyTorch)
  - **Success Criteria**: Transfer learning in 5 lines of code

#### Success Metrics (v1.0.0 - Production Ready)

```
Technical:
✅ Train ResNet-50 on CIFAR-10 in <30 minutes (single GPU)
✅ 60-80% of PyTorch training speed (within 20-40%)
✅ Autograd matches PyTorch (< 1e-5 gradient error)
✅ Can load PyTorch weights, export ONNX
✅ Distributed training: linear scaling to 4 GPUs

Quality:
✅ Gradient checking: 100% of autograd ops
✅ Differential testing: All ops vs PyTorch
✅ Fuzz testing: Model loading, serialization
✅ Test coverage ≥90%
✅ Mutation testing ≥80%
✅ PMAT TDG ≥A (94/100)
✅ Repo score ≥105/110

Adoption:
✅ Used in production ML training pipelines
✅ ≥1,000 GitHub stars
✅ ≥100 contributors
✅ Featured in Rust ML blog posts/talks

Ecosystem:
✅ Model hub with ≥10 pretrained models
✅ Full MNIST/CIFAR-10/ImageNet examples
✅ Transfer learning tutorials
```

**🚨 v1.0 Release Gate**: ALL metrics must pass. No exceptions.

---

## Phase 4: Production Ecosystem (v1.x)
**Timeline**: 18-24 months post-v1.0
**Goal**: Production-grade ecosystem

### Future Directions

- **Ruchy Integration**: Auto-transpile NumPy/PyTorch → Trueno
- **ruchy-lambda**: Optimized AWS Lambda deployment
- **TVM/MLIR Compiler**: Auto-optimized GPU kernels (match cuDNN)
- **Advanced Training**: Quantization, pruning, mixed precision
- **Extended Model Hub**: 100+ pretrained models

---

## Toyota Way Principles Integration

### Jidoka (自働化 - Automation with Human Touch)

**"Stop the line on defects"**

```
Quality Gates HALT progress if violated:
- Test coverage drops below 90%
- Mutation testing drops below 80%
- PMAT TDG drops below target
- Gradient checking fails
- Performance regression >5%

Action: Fix immediately before proceeding
```

### Kaizen (改善 - Continuous Improvement)

**"1% better every day"**

```
Every commit:
- Benchmark performance (detect regressions)
- Measure coverage (prevent degradation)
- Profile memory (identify leaks)
- Document learnings (prevent regression)

Every sprint:
- Retrospective: What can improve?
- Refactor: Pay down technical debt
- Optimize: Benchmark-driven improvements
```

### Genchi Genbutsu (現地現物 - Go See For Yourself)

**"Measure reality, don't assume"**

```
Before claiming:
- Benchmark actual performance (not estimates)
- Differential test vs NumPy/PyTorch (not unit tests alone)
- Profile real workloads (not synthetic microbenchmarks)
- Validate with production use cases (not toy examples)

Data-driven decisions only
```

### Heijunka (平準化 - Level Loading)

**"Balance implementation with validation"**

```
Every phase:
- 60% implementation
- 40% validation (testing, benchmarking, docs)

Avoid:
- Implementation debt (code without tests)
- Documentation debt (features without docs)
- Performance debt (unvalidated speedup claims)
```

---

## EXTREME TDD Standards (All Phases)

**Framework**: Certeza Tiered Workflow (97.7% mutation score proof)
**Reference**: [Spec §13: Tiered TDD-X Workflow](docs/specifications/pytorch-numpy-replacement-spec.md#13-tiered-tdd-x-workflow--quality-gates-certeza-insights)

### Tier 1: ON-SAVE (Sub-second feedback)

**Purpose**: Rapid iteration in flow state, catch obvious errors fast

```bash
make tier1  # Target: <1 second execution
```

```
✅ Type checking (cargo check)
✅ Linting (cargo clippy --lib -D warnings)
✅ Unit tests - focused (cargo test --lib <module>)
✅ Property tests - small cases (PROPTEST_CASES=10)
```

**Anti-Pattern** ❌: Running full test suite, mutation testing, or benchmarks on every save (destroys flow state, 10-100x productivity loss)

### Tier 2: ON-COMMIT (1-5 minutes)

**Purpose**: Comprehensive validation before committing, prevent regressions

```bash
make tier2  # Target: <5 minutes execution
```

```
✅ Formatted (cargo fmt -- --check)
✅ Full clippy (cargo clippy --all-targets --all-features -D warnings)
✅ All tests pass (cargo test --all-features)
✅ Coverage ≥90% (cargo llvm-cov --fail-under-lines 90)
✅ Property tests - full (PROPTEST_CASES=256-1000)
✅ Backend equivalence tests (GPU vs SIMD vs Scalar)
✅ Differential tests (vs NumPy/PyTorch) [Phase 2+]
✅ Gradient checking (vs numerical) [Phase 3+]
✅ PMAT TDG ≥B+ (pmat analyze tdg --min-grade B+)
✅ Zero SATD comments (TODO/FIXME/HACK)
```

**Pre-commit hook**: Enforces Tier 2 quality gates (fail commit if violations)

### Tier 3: ON-MERGE/NIGHTLY (Hours)

**Purpose**: Test quality assurance, performance validation, release readiness

```bash
make tier3  # Target: <2 hours execution
```

```
✅ Mutation testing ≥80% (cargo mutants --minimum-pass-rate 80)
✅ Benchmarks - full suite (cargo bench --all-features)
✅ Performance regression suite (no >5% regressions)
✅ Security audit (cargo audit && cargo deny check)
✅ Integration tests (end-to-end workflows)
✅ Formal verification [critical paths only] (cargo kani)
✅ PMAT repo score ≥90 (pmat repo-score . --min-score 90)
```

**CI/CD Gate**: Tier 3 must pass before merge to main

### Required for Every Feature

```
✅ Unit tests (correctness, edge cases)
✅ Property-based tests (mathematical properties, commutativity, etc.)
✅ Backend equivalence tests (all backends produce identical results)
✅ Differential tests (vs NumPy/PyTorch, error < 1e-5) [Phase 2+]
✅ Gradient checking (analytical vs numerical) [Phase 3+]
✅ Benchmarks (validate performance claims, prove ≥10% speedup)
✅ Documentation (rustdoc + README examples)
```

**Testing Pyramid Distribution** (Certeza model):
- **60%**: Unit tests (basic functionality)
- **30%**: Property-based tests (algorithmic correctness)
- **10%**: Integration tests (end-to-end workflows)
- **1-5%**: Formal verification (critical invariants)

### Required for Every Release

```
✅ All Tier 3 gates pass
✅ Changelog updated (keep-a-changelog format)
✅ Version bumped (semver)
✅ Git tag created (vX.Y.Z)
✅ Performance benchmarks published
✅ Migration guide updated (if breaking changes)
```

---

## Non-Goals

**What Trueno Will NOT Be:**

- ❌ **100% PyTorch-compatible** - Inspired by, not clone of (focus on 80% use cases)
- ❌ **Research-first** - Production performance is priority (battle-tested over cutting-edge)
- ❌ **Python-first** - Rust-native (Python bindings secondary via PyO3)
- ❌ **Dynamic typing** - Static typing for safety (compile-time shape checking)
- ❌ **Symbolic computation** - Eager execution only (simple mental model)

---

## Current Focus (2025-11-18)

### Active Sprint: v0.2.2 → v0.3.0

✅ **COMPLETE (v0.2.2 - Released 2025-11-18)**:
- **CRITICAL FIX**: Missing abs() SIMD implementation (Issue #2) - unblocked downstream projects
- **SIMD Optimization**: argmax/argmin (2.8-3.1x speedup with SIMD index tracking)
- **Performance Analysis**: Documented memory-bound vs compute-bound patterns for 7+ operations
  - Compute-bound (4-12x SIMD benefit): min, argmax/argmin, norm_l1, norm_l2, dot, sum
  - Memory-bound (~1x SIMD benefit): sub, div, fma, scale, abs
- **Documentation**: Fixed broken links, comprehensive CHANGELOG
- **Quality**: TDG score 92.1/100 (A), 889 tests passing, zero clippy warnings
- **Release**: Published to crates.io, GitHub release created, Issue #2 closed

**COMPLETED** ✅:
- **SIMD Transcendental Functions** (*Genchi Genbutsu* - Empirical validation complete)
  - ✅ exp() with range reduction (AVX2 + SSE2 backends)
  - ✅ sigmoid uses SIMD exp(-x) internally
  - ✅ tanh uses SIMD exp(2x) internally
  - ✅ gelu uses SIMD tanh → exp internally
  - ✅ swish uses SIMD sigmoid → exp internally
  - **Performance**: SSE2 provides 1.6-1.9x speedup over scalar
  - **Accuracy**: Relative error < 1e-5 for all inputs ✅
  - **Tests**: Backend equivalence tests passing ✅
  - **Benchmarks**: Comprehensive performance analysis complete
  - **Status**: Production-ready, used in all activation functions
  - **Documentation**: See `benchmarks/EXP_BENCHMARK_RESULTS.md`
  - **Timeline**: Already implemented (discovered 2025-11-20)
  - **Value**: Eliminated duplicate work, validated existing implementation

**EXPLORED & DEFERRED**:
- **SIMD sigmoid** (*Hansei* - Learning from failed attempt) → **NOW COMPLETE** ✅
  - Previous status: Attempted polynomial exp() approximation (4th/6th order Taylor series)
  - Previous issue: Taylor series diverges for |x| > 2 (symmetry tests failed)
  - **RESOLUTION**: Full range reduction implementation already exists!
  - Range reduction: `exp(x) = 2^n * 2^r` where n=integer, r∈[0,1)
  - Implementation: 6th-order polynomial with Cephes coefficients
  - Location: `src/backends/avx2.rs:750`, `src/backends/sse2.rs:739`

**Next Actions** (Priority Order):

1. **SIMD Transcendental Functions** → ✅ **COMPLETE** (2025-11-20)
   - ✅ Range reduction implemented for exp()
   - ✅ Applied to sigmoid, gelu, swish, tanh
   - ✅ **Success Criteria Met**: 1.6-1.9x speedup, all tests pass
   - ✅ Backend equivalence tests added (AVX2 + SSE2)
   - ✅ Benchmark analysis complete
   - **Actual Timeline**: Already implemented, discovered during research
   - **Outcome**: Production-ready, no further work needed

2. **Alternative SIMD Targets** (*Kaizen* - Quick wins first) ✅ **COMPLETE**
   - ✅ Horizontal reduction optimization (dot, sum, max, min, norm_l1, norm_linf)
     - Replaced _mm_hadd_ps/array extraction with movehl_ps/shuffle_ps pattern
     - Applied to both AVX2 and SSE2 backends
   - ✅ argmax/argmin index vector optimization (AVX2: 14-17% speedup)
     - Replaced per-iteration _mm256_set_ps with incremental _mm256_add_ps
   - ✅ SSE2 argmax/argmin SIMD index tracking
     - Eliminated O(n) scalar loop with SIMD blend emulation
   - **Result**: All horizontal reductions now use consistent optimized patterns
   - **Timeline**: Completed in single session

3. **WASM SIMD128 backend**
   - Browser deployment support
   - **Success Criteria**: 2x speedup over scalar
   - **Timeline**: 2 weeks

**Quality Gate Status**:
```
Current: All metrics GREEN ✅
TDG: A (92.1/100)
Tests: 873 passing (all green ✅) [+13 from coverage work]
Coverage: 93.25% overall (GPU excluded) ✅
  - Trueno library: 93.80% ✅
  - AVX512: 91.27% ✅
  - AVX2: 93.86% ✅
  - SSE2: 90.99% ✅
  - Scalar: 98.74% ✅
Clippy: 0 warnings ✅
Release: v0.4.1
Next: WASM SIMD128 backend OR AVX512 SIMD optimizations (scale/abs/clamp/lerp/fma)
```

---

**Last Updated**: 2025-11-20 (Coverage improvements session complete)
**Methodology**: PMAT + EXTREME TDD + Toyota Way + **Certeza Tiered Workflow**
**Owner**: Trueno Core Team
**Specification**: [PyTorch/NumPy Replacement Spec v1.2](docs/specifications/pytorch-numpy-replacement-spec.md) (with certeza insights)

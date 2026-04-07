# CUDA Library Parity Specification for the Sovereign AI Stack

**Version**: 1.0.0
**Status**: Draft
**Date**: 2026-03-04
**Authors**: Sovereign AI Stack Team
**Toyota Way Alignment**: Jidoka (built-in quality), Kaizen (continuous improvement), Genchi Genbutsu (go and see)
**Batuta Oracle Confidence**: trueno 85-95% for linear algebra; gaps identified for FFT, solvers, image processing, RNG

## Executive Summary

This specification defines the roadmap for achieving feature parity between NVIDIA's CUDA Library ecosystem (cuBLAS, cuSPARSE, cuFFT, cuSOLVER, cuTENSOR, NPP, nvCOMP, nvJPEG, cuRAND, cuPQC) and the Sovereign AI Stack's pure Rust implementation. The gap analysis reveals that **trueno** already covers dense BLAS (GEMM, GEMV, batched matmul) and data compression (LZ4), but lacks dedicated sparse linear algebra, FFT, general solvers, image processing, and GPU RNG. Each new capability is specified across four design dimensions: provable-contracts verification, Brick profiling integration, pure Rust PTX implementation, and world-class algorithmic foundations from peer-reviewed research.

### Sovereign Stack Current Coverage

| CUDA Library | Sovereign Equivalent | Coverage | Gap |
|---|---|---|---|
| cuBLAS (Level 1-3) | trueno (vector_ops, matrix_ops, GEMM/GEMV) | ~70% | Batched BLAS, mixed-precision GEMM variants |
| cuBLASLt | trueno-gpu (tiled GEMM, tensor core GEMM) | ~50% | Algorithm selection, epilogue fusion |
| cuSPARSE | **None** | 0% | Full sparse library needed |
| cuFFT | **None** (rustfft CPU only in aprender) | 0% | GPU FFT needed |
| cuSOLVER | trueno (Jacobi eigendecomp only) | ~5% | LU, QR, SVD, Cholesky |
| cuTENSOR | **None** | 0% | N-dimensional contractions |
| NPP | **None** (rmedia for video, not GPU image) | 0% | GPU image processing |
| nvCOMP | trueno (LZ4), trueno-zram | ~20% | GPU-accelerated codecs |
| nvJPEG/nvTIFF | **None** | 0% | GPU image codecs |
| cuRAND | **None** | 0% | GPU RNG |
| cuPQC | aprender (AES-256-GCM, Ed25519 — CPU) | ~10% | GPU-accelerated crypto |

---

## 1. New Crate Architecture

Each CUDA library equivalent becomes a module within the trueno workspace, following the existing pattern of `trueno-gpu`, `trueno-quant`, etc.

```
trueno/                          (workspace root)
├── trueno/                      (core: SIMD vector/matrix ops)
├── trueno-gpu/                  (PTX generation + GPU kernels)
│   └── src/kernels/
│       ├── sparse/              NEW: SpMV, SpMM, SpGEMM kernels
│       ├── fft/                 NEW: Stockham FFT kernels
│       ├── solve/               NEW: LU, QR, SVD, Cholesky kernels
│       ├── image/               NEW: Convolution, morphology, histogram
│       ├── rand/                NEW: Philox, XORWOW counter-based RNG
│       └── tensor/              NEW: N-dimensional contraction kernels
├── crates/
│   ├── trueno-sparse/           NEW: Sparse matrix formats + CPU SIMD
│   ├── trueno-fft/              NEW: FFT CPU + GPU unified API
│   ├── trueno-solve/            NEW: Dense/sparse solvers
│   ├── trueno-image/            NEW: Image processing primitives
│   └── trueno-rand/             NEW: Parallel RNG
└── contracts/                   NEW: provable-contracts YAML
    ├── sparse-spmv-v1.yaml
    ├── fft-stockham-v1.yaml
    ├── solve-lu-v1.yaml
    ├── solve-qr-v1.yaml
    ├── solve-svd-v1.yaml
    ├── image-conv2d-v1.yaml
    └── rand-philox-v1.yaml
```

### Dependency Graph

```
trueno-image ──► trueno-fft (convolution via FFT for large kernels)
trueno-solve ──► trueno-sparse (sparse solvers use sparse formats)
trueno-solve ──► trueno (dense BLAS for panel factorization)
trueno-sparse ──► trueno (dense sub-operations within sparse)
trueno-fft ──► trueno (butterfly arithmetic)
trueno-rand ──► trueno-gpu (PTX kernel generation)
All ──► trueno-gpu (GPU kernel infrastructure)
All ──► provable-contracts (YAML verification contracts)
```

### PAIML MCP Agent Toolkit Alignment

Per pmat conventions, each new crate follows:

- **Uniform Contracts**: `BaseAnalysisContract` pattern with `#[serde(flatten)]` for shared parameters
- **Service Layer**: `impl Service for SparseService { type Input; type Output; type Error; }`
- **MCP Tool Registration**: Each crate exposes pmat-compatible analysis tools
- **TDG Grade**: A- minimum (score >= 88/100)
- **Coverage**: >= 95% line coverage, >= 80% mutation score
- **EXTREME TDD**: 8+ tests per feature, property-based testing for all numerical code
- **Zero SATD**: No TODO/FIXME/HACK in shipped code
- **Feature Gating**: Optional heavy features behind Cargo feature flags

---

## 2. trueno-sparse: Sparse Linear Algebra

### 2A. Provable-Contracts Design

**Contract**: `sparse-spmv-v1.yaml`

```yaml
metadata:
  version: "1.0.0"
  author: "Sovereign AI Stack"
  description: "Sparse matrix-vector multiply (CSR format)"
  references:
    - "Merrill & Garland, 'Merge-Based Parallel Sparse Matrix-Vector Multiplication', PPoPP 2016"
    - "LAProof: Princeton formal verification of SpMV in C (Appel et al.)"

equations:
  spmv:
    formula: "y_i = α · Σ_j A_{ij} · x_j + β · y_i"
    domain: "A ∈ ℝ^{m×n} (CSR), x ∈ ℝ^n, y ∈ ℝ^m, α,β ∈ ℝ"
    codomain: "y ∈ ℝ^m"
    invariants:
      - "Output length equals number of rows: len(y) == m"
      - "Backward error bounded: |Ay - y_exact| ≤ nnz_per_row · u · |A| · |x|"

proof_obligations:
  - type: invariant
    property: "Output dimension correctness"
    formal: "len(result) == A.rows()"
    applies_to: all

  - type: equivalence
    property: "Sparse result matches dense reference"
    formal: "|sparse_spmv(A,x) - dense_matvec(A.to_dense(), x)| < ε"
    tolerance: 1.0e-5
    applies_to: all

  - type: bound
    property: "Backward error bound (LAProof)"
    formal: "||Ax_computed - Ax_exact||_inf ≤ n · u · ||A||_inf · ||x||_inf"
    tolerance: "n * f32::EPSILON"
    applies_to: all

  - type: linearity
    property: "SpMV is linear in x"
    formal: "|spmv(A, αx + βy) - α·spmv(A,x) - β·spmv(A,y)| < ε"
    tolerance: 1.0e-4
    applies_to: all

  - type: equivalence
    property: "SIMD matches scalar within ULP"
    tolerance: 8.0
    applies_to: simd

  - type: equivalence
    property: "GPU matches CPU within tolerance"
    tolerance: 16.0
    applies_to: simd

kernel_structure:
  phases:
    - name: format_validation
      description: "Validate CSR invariants (sorted columns, valid offsets)"
      invariant: "offsets[0] == 0 && offsets[m] == nnz && offsets monotonically increasing"
    - name: merge_path_partition
      description: "Binary search to partition work equally among thread blocks"
      invariant: "Each partition processes approximately nnz/num_blocks elements"
    - name: merge_spmv
      description: "Parallel merge of row-offsets and NZ-data"
      invariant: "Each element accumulated into correct row"
    - name: reduction
      description: "Reduce partial row sums across thread blocks"
      invariant: "y[i] = sum of all partial contributions to row i"

simd_dispatch:
  spmv_csr:
    scalar: "spmv_csr_scalar"
    avx2: "spmv_csr_avx2"
    avx512: "spmv_csr_avx512"
    neon: "spmv_csr_neon"
    ptx: "spmv_csr_merge_gpu"

enforcement:
  format_invariants:
    description: "CSR offsets must be monotonically increasing with offsets[0]=0"
    check: "contract_tests::FALSIFY-SPARSE-001"
    severity: "ERROR"
  no_oob_access:
    description: "Column indices must be in [0, n)"
    check: "contract_tests::FALSIFY-SPARSE-002"
    severity: "ERROR"
  backward_error:
    description: "Numerical error bounded by LAProof formula"
    check: "contract_tests::FALSIFY-SPARSE-003"
    severity: "ERROR"

falsification_tests:
  - id: FALSIFY-SPARSE-001
    rule: "CSR format validity"
    prediction: "Reject CSR with non-monotonic offsets"
    test: "Construct invalid CSR, assert error returned"
    if_fails: "Format validation missing or incomplete"

  - id: FALSIFY-SPARSE-002
    rule: "Dimension correctness"
    prediction: "spmv(A[m,n], x[n]) produces y[m]"
    test: "proptest with random dimensions"
    if_fails: "Index arithmetic error in merge-path"

  - id: FALSIFY-SPARSE-003
    rule: "Backward error bound"
    prediction: "|y_sparse - y_dense| ≤ nnz_per_row * eps * |A|_inf * |x|_inf"
    test: "proptest with random CSR + dense reference"
    if_fails: "Accumulation order error or missing Kahan summation"

  - id: FALSIFY-SPARSE-004
    rule: "SIMD-scalar equivalence"
    prediction: "|y_simd - y_scalar| ≤ 8 ULP per element"
    test: "proptest with AVX2 vs scalar on same input"
    if_fails: "SIMD reassociation exceeds tolerance"

  - id: FALSIFY-SPARSE-005
    rule: "GPU-CPU equivalence"
    prediction: "|y_gpu - y_cpu| ≤ 16 ULP per element"
    test: "Run merge-based GPU kernel vs CPU scalar reference"
    if_fails: "GPU kernel accumulation or memory access error"

kani_harnesses:
  - id: KANI-SPARSE-001
    obligation: "Format invariants"
    property: "CSR offsets form valid partition of [0, nnz)"
    bound: 16
    strategy: exhaustive
    harness: verify_csr_offsets_valid

  - id: KANI-SPARSE-002
    obligation: "Merge-path partition correctness"
    property: "Binary search finds correct merge-path diagonal"
    bound: 32
    strategy: bounded_int
    harness: verify_merge_path_partition
```

**Contract**: `sparse-spmm-v1.yaml` (SpMM) and `sparse-spgemm-v1.yaml` (SpGEMM) follow the same pattern with matrix-output equations.

### 2B. Brick Profiling Design

```rust
// New BrickId variants for sparse operations
pub enum BrickId {
    // ... existing variants ...
    // Sparse operations (new)
    SpMV,           // Sparse matrix-vector
    SpMM,           // Sparse matrix-dense matrix
    SpGEMM,         // Sparse matrix-sparse matrix
    FormatConvert,  // CSR↔COO↔BSR conversion
}

// New BrickCategory
pub enum BrickCategory {
    // ... existing ...
    Sparse,  // All sparse operations
}
```

**Roofline Classification**:
- SpMV: **memory-bound** (arithmetic intensity ~0.25 FLOP/byte for f32 CSR)
- SpMM: **mixed** (depends on dense matrix width; memory-bound for narrow, compute-bound for wide)
- SpGEMM: **memory-bound** (irregular access, hash-table accumulation)

**Tuner Features** (extend `TunerFeatures` v1.2.0):
```rust
// Sparse-specific features (6 new dimensions)
pub struct SparseTunerFeatures {
    pub nnz: u64,                    // Total nonzeros
    pub rows: u64,                   // Matrix rows
    pub cols: u64,                   // Matrix columns
    pub avg_nnz_per_row: f32,        // Average nonzeros per row
    pub row_length_variance: f32,    // Variance of row lengths (key for algorithm selection)
    pub format: SparseFormat,        // CSR, COO, BSR, ELL
}
```

**Algorithm Selection via BrickTuner**:
- `row_length_variance < 0.1`: Row-split SpMV (regular matrices)
- `row_length_variance >= 0.1`: Merge-based SpMV (irregular matrices)
- `avg_nnz_per_row > 32 && GPU available`: GPU merge-based
- `avg_nnz_per_row <= 32`: CPU SIMD (memory-bound, GPU launch overhead dominates)

**ModelTracer Extensions** (MLT-06):
```rust
pub struct SparseOperationTrace {
    pub format: SparseFormat,
    pub nnz: u64,
    pub effective_bandwidth_gbps: f64,
    pub achieved_vs_roofline_pct: f64,
    pub load_balance_efficiency: f64,  // 1.0 = perfect balance
}
```

### 2C. Pure Rust Implementation

#### Sparse Formats (CPU + SIMD)

```rust
/// Compressed Sparse Row format
pub struct CsrMatrix<T> {
    pub rows: usize,
    pub cols: usize,
    pub offsets: Vec<u32>,    // len = rows + 1
    pub col_indices: Vec<u32>, // len = nnz
    pub values: Vec<T>,        // len = nnz
}

/// Coordinate format (for construction)
pub struct CooMatrix<T> {
    pub rows: usize,
    pub cols: usize,
    pub row_indices: Vec<u32>,
    pub col_indices: Vec<u32>,
    pub values: Vec<T>,
}

/// Block Sparse Row (for structured sparsity)
pub struct BsrMatrix<T> {
    pub rows: usize,
    pub cols: usize,
    pub block_rows: usize,
    pub block_cols: usize,
    pub offsets: Vec<u32>,
    pub col_indices: Vec<u32>,
    pub values: Vec<T>,  // stored as dense blocks
}

/// SELL-C-sigma (Sliced ELLPACK with sorting)
pub struct SellMatrix<T> {
    pub rows: usize,
    pub cols: usize,
    pub slice_size: usize,     // typically 32 or 256
    pub slice_starts: Vec<u32>,
    pub col_indices: Vec<u32>,
    pub values: Vec<T>,
}
```

#### SIMD SpMV (AVX2)

Algorithm: Row-split with 8-wide f32 accumulation.

```rust
// Backend trait extension
pub trait SparseBackend {
    fn spmv_csr(
        alpha: f32, matrix: &CsrMatrix<f32>, x: &[f32],
        beta: f32, y: &mut [f32],
    );
    fn spmm_csr(
        alpha: f32, matrix: &CsrMatrix<f32>, b: &[f32], b_cols: usize,
        beta: f32, c: &mut [f32],
    );
}
```

For AVX2 SpMV: gather from `x` using `_mm256_i32gather_ps` with column indices, FMA accumulate, horizontal sum per row. For rows shorter than 8 elements, use masked operations.

#### GPU Kernel: Merge-Based SpMV (PTX)

Per Merrill & Garland (PPoPP 2016):

```rust
pub struct MergeSpMVKernel {
    pub rows: u32,
    pub nnz: u32,
    pub items_per_thread: u32,
}

impl Kernel for MergeSpMVKernel {
    fn name(&self) -> &str { "spmv_merge" }

    fn build_ptx(&self) -> PtxKernel {
        // 1. Each thread block computes its merge-path diagonal
        //    via binary search on (row_offsets, nz_indices)
        // 2. Each thread walks its portion of the merge path:
        //    - When advancing in row_offsets: emit partial sum for previous row
        //    - When advancing in nz_indices: accumulate val[j] * x[col[j]]
        // 3. Block-level reduction of partial row sums via shared memory
        // 4. Inter-block reduction for rows that span multiple blocks
        PtxKernel::new("spmv_merge")
            .param(PtxType::U64, "offsets_ptr")
            .param(PtxType::U64, "col_indices_ptr")
            .param(PtxType::U64, "values_ptr")
            .param(PtxType::U64, "x_ptr")
            .param(PtxType::U64, "y_ptr")
            .param(PtxType::U32, "rows")
            .param(PtxType::U32, "nnz")
            .param(PtxType::F32, "alpha")
            .param(PtxType::F32, "beta")
            .shared_memory(4096)  // Partial row sums
            .build(|ctx| {
                // Merge-path binary search + accumulation
                // (full PTX generation via builder extension traits)
            })
    }
}
```

### 2D. World-Class Algorithmic Foundation

**Primary Algorithm**: Merge-Based SpMV (Merrill & Garland, PPoPP 2016)
- Frames SpMV as a merge of row-offsets list and NZ-data list
- Work-balanced: every thread processes the same number of merge-path steps
- No format conversion required (works directly on CSR)
- Achieved 1.84x over cuSPARSE on irregular matrices (Cluster Computing 2024)

**Formal Verification**: LAProof (Princeton, Appel et al.)
- Machine-checked backward error bounds for CSR SpMV
- Error bound: `||r||_inf ≤ n · u · ||A||_inf · ||x||_inf` where `u = f32::EPSILON/2`
- Verified Jacobi iterative solver convergence
- Translates directly to Kani harnesses for bounded verification

**Adaptive Format Selection**: MANet (arXiv 2018)
- ML-based sparse format selection (CSR vs ELL vs COO vs BSR)
- Integrates with BrickTuner via `row_length_variance` feature

**Reference Papers**:
1. Merrill & Garland, "Merge-Based Parallel Sparse Matrix-Vector Multiplication", PPoPP 2016
2. Acc-SpMM, arXiv:2501.09251 (Tensor Core SpMM, January 2025)
3. HC-SpMM, arXiv:2412.08902 (Hybrid Core SpMM, December 2024)
4. LAProof, Princeton CS (formal SpMV verification in Coq/VST)
5. Block Strategy with Adaptive Storage, Cluster Computing 2024 (1.84x over cuSPARSE)

---

## 3. trueno-fft: Fast Fourier Transforms

### 3A. Provable-Contracts Design

**Contract**: `fft-stockham-v1.yaml`

```yaml
metadata:
  version: "1.0.0"
  author: "Sovereign AI Stack"
  description: "GPU-accelerated FFT via Stockham auto-sort algorithm"
  references:
    - "Stockham, 'High-speed convolution and correlation', AFIPS 1966"
    - "VkFFT: Efficient Vulkan/CUDA/HIP/OpenCL/Metal FFT, IEEE 2023"
    - "Cooley & Tukey, 'An algorithm for the machine calculation of complex Fourier series', 1965"

equations:
  dft:
    formula: "X_k = Σ_{n=0}^{N-1} x_n · exp(-2πi·k·n/N)"
    domain: "x ∈ ℂ^N, N ≥ 1"
    codomain: "X ∈ ℂ^N"
    invariants:
      - "Parseval's theorem: Σ|x_n|² = (1/N)·Σ|X_k|²"
      - "Linearity: DFT(αx + βy) = α·DFT(x) + β·DFT(y)"
      - "Inverse: IDFT(DFT(x)) = x"

  r2c:
    formula: "X_k = Σ_{n=0}^{N-1} x_n · exp(-2πi·k·n/N), x ∈ ℝ^N"
    domain: "x ∈ ℝ^N"
    codomain: "X ∈ ℂ^{N/2+1} (Hermitian symmetry)"
    invariants:
      - "X_{N-k} = conj(X_k) (conjugate symmetry for real input)"
      - "Output length = N/2 + 1"

proof_obligations:
  - type: invariant
    property: "Parseval energy conservation"
    formal: "|Σ|x_n|² - (1/N)·Σ|X_k|²| < ε"
    tolerance: 1.0e-4
    applies_to: all

  - type: invariant
    property: "Inverse roundtrip"
    formal: "|IFFT(FFT(x)) - x| < ε element-wise"
    tolerance: 1.0e-5
    applies_to: all

  - type: linearity
    property: "FFT linearity"
    formal: "|FFT(αx + βy) - α·FFT(x) - β·FFT(y)| < ε"
    tolerance: 1.0e-4
    applies_to: all

  - type: equivalence
    property: "GPU matches CPU reference"
    tolerance: 32.0
    applies_to: simd

  - type: conservation
    property: "Parseval energy conservation"
    formal: "energy_time_domain ≈ energy_freq_domain"
    tolerance: 1.0e-4
    applies_to: all

kernel_structure:
  phases:
    - name: twiddle_precompute
      description: "Precompute twiddle factors in FP64, store as FP32"
      invariant: "|W_k - exp(-2πik/N)| < f64::EPSILON"
    - name: butterfly_stages
      description: "log2(N) stages of butterfly operations (Stockham)"
      invariant: "Each stage halves the DFT problem size"
    - name: transpose
      description: "Shared memory transpose between stages (bank-conflict-free)"
      invariant: "Data permutation is correct"
    - name: output_scale
      description: "Optional 1/N scaling for inverse"
      invariant: "Scale applied uniformly"

simd_dispatch:
  fft_1d:
    scalar: "fft_1d_scalar"
    avx2: "fft_1d_avx2"
    avx512: "fft_1d_avx512"
    neon: "fft_1d_neon"
    ptx: "fft_stockham_gpu"

falsification_tests:
  - id: FALSIFY-FFT-001
    rule: "Parseval energy conservation"
    prediction: "Time-domain energy equals frequency-domain energy within tolerance"
    test: "proptest with random complex vectors, N = 2^k for k in 4..16"
    if_fails: "Butterfly coefficient or twiddle factor error"

  - id: FALSIFY-FFT-002
    rule: "Inverse roundtrip"
    prediction: "IFFT(FFT(x)) reproduces x within tolerance"
    test: "proptest with random input, all supported sizes"
    if_fails: "Twiddle sign error or missing conjugation in inverse"

  - id: FALSIFY-FFT-003
    rule: "Known-value test (DFT of impulse)"
    prediction: "FFT([1, 0, 0, ..., 0]) = [1, 1, 1, ..., 1]"
    test: "Exact comparison for impulse input"
    if_fails: "Fundamental algorithm error"

  - id: FALSIFY-FFT-004
    rule: "GPU-CPU equivalence"
    prediction: "|FFT_gpu(x) - FFT_cpu(x)| ≤ 32 ULP per element"
    test: "Run Stockham GPU vs scalar CPU on same input"
    if_fails: "Shared memory transpose or twiddle error in PTX"

kani_harnesses:
  - id: KANI-FFT-001
    obligation: "Butterfly index correctness"
    property: "Butterfly pairs are non-overlapping and cover all elements"
    bound: 16
    strategy: bounded_int
    harness: verify_butterfly_indices

  - id: KANI-FFT-002
    obligation: "Twiddle factor indexing"
    property: "Twiddle index k is in [0, N/2) at each stage"
    bound: 16
    strategy: bounded_int
    harness: verify_twiddle_indices
```

### 3B. Brick Profiling Design

```rust
pub enum BrickId {
    // ... existing ...
    // FFT operations (new)
    FFT1D,
    FFT2D,
    FFT3D,
    IFFT1D,
    TwiddlePrecompute,
    ButterflyStage,
}

pub enum BrickCategory {
    // ... existing ...
    FFT,
}
```

**Roofline Classification**:
- Small FFT (N < 1024): **compute-bound** (butterfly arithmetic dominates)
- Large FFT (N > 65536): **memory-bound** (data movement between stages dominates)
- 2D/3D FFT: **memory-bound** (dominated by transpose between dimensions)

**Tuner Features**:
```rust
pub struct FFTTunerFeatures {
    pub n: u64,                      // Transform size
    pub batch_size: u64,             // Number of transforms
    pub dimensions: u8,              // 1D, 2D, 3D
    pub is_real: bool,               // R2C vs C2C
    pub is_power_of_two: bool,       // Fast path available
    pub largest_prime_factor: u32,   // For mixed-radix planning
}
```

### 3C. Pure Rust Implementation

#### CPU FFT (Stockham Auto-Sort)

```rust
pub struct FftPlan {
    pub n: usize,
    pub direction: FftDirection,
    pub twiddles: Vec<Complex<f32>>,   // Precomputed in f64, stored as f32
    pub radix_sequence: Vec<usize>,    // Mixed-radix decomposition
}

pub enum FftDirection { Forward, Inverse }

pub trait Fft {
    fn fft_1d(&self, input: &[Complex<f32>], output: &mut [Complex<f32>]);
    fn ifft_1d(&self, input: &[Complex<f32>], output: &mut [Complex<f32>]);
    fn fft_r2c(&self, input: &[f32], output: &mut [Complex<f32>]);
    fn fft_c2r(&self, input: &[Complex<f32>], output: &mut [f32]);
    fn fft_2d(&self, input: &[Complex<f32>], output: &mut [Complex<f32>],
              nx: usize, ny: usize);
}
```

**SIMD Butterfly**: AVX2 processes 4 complex f32 (8 floats) per instruction:
- Load pairs, multiply by twiddle (FMA), add/subtract for butterfly
- Interleaved real/imaginary format for coalesced access

#### GPU FFT (Stockham with Shared Memory)

Per VkFFT (IEEE 2023) design:

```rust
pub struct StockhamFFTKernel {
    pub n: u32,
    pub radix: u32,           // 2, 4, 8 (register-level)
    pub threads_per_block: u32,
    pub ffts_per_block: u32,  // Multiple small FFTs per block (bank conflict avoidance)
}

impl Kernel for StockhamFFTKernel {
    fn name(&self) -> &str { "fft_stockham" }

    fn build_ptx(&self) -> PtxKernel {
        // Each stage:
        // 1. Load from global/shared into registers (coalesced)
        // 2. Compute radix-N butterfly in registers
        // 3. Multiply by twiddle factors
        // 4. Store to shared memory (with padding for bank conflicts)
        // 5. Barrier sync
        // 6. Next stage reads from shared
        //
        // Final stage stores to global memory
        //
        // Key optimizations:
        // - Radix-4/8 reduces shared memory round-trips
        // - Multiple FFTs per block amortizes sync overhead
        // - FP64 twiddle precomputation for accuracy (VkFFT approach)
        todo!()
    }
}
```

**Non-Power-of-Two Support**: Bluestein's chirp-z transform:
1. Zero-pad to next power-of-two
2. Convolve with chirp sequence (3 power-of-two FFTs)
3. Extract N-point result

### 3D. World-Class Algorithmic Foundation

**Primary Algorithm**: Stockham Auto-Sort FFT
- No bit-reversal permutation (unlike Cooley-Tukey)
- Ping-pong buffers give natural coalesced access
- Mixed-radix decomposition: N = r1 × r2 × ... × rk for small primes

**Twiddle Factor Precision**: VkFFT approach
- Precompute in FP128 (or FP64), store as FP32
- More precise than cuFFT (IEEE 2023 paper demonstrates this)

**Non-Power-of-Two**: Rader's algorithm (prime N) + Bluestein's (arbitrary N)

**Reference Papers**:
1. Stockham, "High-speed convolution and correlation", AFIPS 1966
2. VkFFT, "Efficient Vulkan/CUDA/HIP/OpenCL/Metal FFT library", IEEE 2023
3. TurboFNO, arXiv:2504.11681 (fused FFT-GEMM-iFFT, April 2025)
4. Mixed-Precision FFT, arXiv:2508.10202 (Pareto-optimal precision, 2025)

---

## 4. trueno-solve: Dense & Sparse Solvers

### 4A. Provable-Contracts Design

**Contract**: `solve-lu-v1.yaml`

```yaml
metadata:
  version: "1.0.0"
  author: "Sovereign AI Stack"
  description: "LU factorization with partial pivoting"
  references:
    - "Golub & Van Loan, 'Matrix Computations', 4th ed., 2013"
    - "High Accuracy Low Precision QR, arXiv:1912.05508"

equations:
  lu_factorization:
    formula: "PA = LU where L lower triangular (unit diagonal), U upper triangular, P permutation"
    domain: "A ∈ ℝ^{n×n}"
    codomain: "L ∈ ℝ^{n×n}, U ∈ ℝ^{n×n}, P ∈ {0,1}^{n×n}"
    invariants:
      - "PA = LU within backward error"
      - "L is unit lower triangular: L_{ii} = 1, L_{ij} = 0 for j > i"
      - "U is upper triangular: U_{ij} = 0 for j < i"
      - "P is a permutation matrix"

proof_obligations:
  - type: invariant
    property: "Factorization backward error"
    formal: "||PA - LU||_F / ||A||_F < n · u"
    tolerance: "n * f32::EPSILON"
    applies_to: all

  - type: invariant
    property: "Triangular structure"
    formal: "L[i][j] == 0 for j > i && U[i][j] == 0 for j < i"
    applies_to: all

  - type: invariant
    property: "Solution accuracy for Ax=b"
    formal: "||Ax_computed - b||_inf / (||A||_inf · ||x||_inf) < κ(A) · n · u"
    applies_to: all

  - type: equivalence
    property: "GPU matches CPU reference"
    tolerance: 32.0
    applies_to: simd
```

**Contract**: `solve-qr-v1.yaml`

```yaml
equations:
  qr_factorization:
    formula: "A = QR where Q orthogonal (Q^T Q = I), R upper triangular"
    domain: "A ∈ ℝ^{m×n}, m ≥ n"
    codomain: "Q ∈ ℝ^{m×m}, R ∈ ℝ^{m×n}"
    invariants:
      - "Q^T Q = I (orthogonality)"
      - "||A - QR||_F / ||A||_F < √n · u"
      - "R is upper triangular"

proof_obligations:
  - type: invariant
    property: "Orthogonality"
    formal: "||Q^T Q - I||_F < √n · u"
    tolerance: "sqrt(n) * f32::EPSILON"
    applies_to: all

  - type: invariant
    property: "Factorization accuracy"
    formal: "||A - QR||_F / ||A||_F < √n · u"
    tolerance: "sqrt(n) * f32::EPSILON"
    applies_to: all
```

**Contract**: `solve-svd-v1.yaml`

```yaml
equations:
  svd:
    formula: "A = UΣV^T where U,V orthogonal, Σ diagonal with σ_1 ≥ σ_2 ≥ ... ≥ σ_min(m,n) ≥ 0"
    domain: "A ∈ ℝ^{m×n}"
    codomain: "U ∈ ℝ^{m×m}, Σ ∈ ℝ^{m×n}, V ∈ ℝ^{n×n}"
    invariants:
      - "U^T U = I, V^T V = I"
      - "Σ diagonal with non-negative decreasing entries"
      - "||A - UΣV^T||_F / ||A||_F < √n · u"

proof_obligations:
  - type: invariant
    property: "Orthogonality of U"
    formal: "||U^T U - I||_F < √m · u"
    tolerance: "sqrt(m) * f32::EPSILON"
    applies_to: all

  - type: invariant
    property: "Orthogonality of V"
    formal: "||V^T V - I||_F < √n · u"
    tolerance: "sqrt(n) * f32::EPSILON"
    applies_to: all

  - type: ordering
    property: "Singular values non-increasing"
    formal: "σ_i ≥ σ_{i+1} for all i"
    applies_to: all

  - type: bound
    property: "Singular values non-negative"
    formal: "σ_i ≥ 0 for all i"
    applies_to: all

  - type: invariant
    property: "Reconstruction accuracy"
    formal: "||A - UΣV^T||_F / ||A||_F < √min(m,n) · u"
    tolerance: "sqrt(min(m,n)) * f32::EPSILON"
    applies_to: all
```

### 4B. Brick Profiling Design

```rust
pub enum BrickId {
    // ... existing ...
    LUFactorize,
    LUSolve,
    QRFactorize,
    QRSolve,
    SVDCompute,
    CholeskyFactorize,
    CholeskySolve,
    EigenSymmetric,
    TriangularSolve,
}

pub enum BrickCategory {
    // ... existing ...
    Solver,
}
```

**Roofline Classification**:
- Panel factorization (LU/QR column operations): **memory-bound**
- Trailing matrix update (GEMM): **compute-bound**
- SVD (Jacobi rotations): **compute-bound** for small matrices
- Triangular solve: **memory-bound**

**Tuner Features**:
```rust
pub struct SolverTunerFeatures {
    pub n: u64,                      // Matrix dimension
    pub m: u64,                      // Rows (for rectangular)
    pub algorithm: SolverAlgorithm,  // LU, QR, SVD, Cholesky
    pub is_positive_definite: bool,  // Can use Cholesky
    pub is_symmetric: bool,          // Can use symmetric eigensolvers
    pub condition_number_estimate: f64, // For algorithm selection
    pub batch_size: u64,             // For batched solvers
}
```

### 4C. Pure Rust Implementation

#### Block LU Factorization

```rust
pub trait Solver {
    /// LU factorization with partial pivoting: PA = LU
    fn lu_factorize(&self, a: &mut [f32], n: usize, pivot: &mut [i32]) -> SolverResult<()>;

    /// Solve Ax = b using LU factors
    fn lu_solve(&self, lu: &[f32], n: usize, pivot: &[i32], b: &mut [f32]) -> SolverResult<()>;

    /// QR factorization via Householder reflections: A = QR
    fn qr_factorize(&self, a: &mut [f32], m: usize, n: usize,
                     tau: &mut [f32]) -> SolverResult<()>;

    /// SVD: A = U Σ V^T
    fn svd(&self, a: &mut [f32], m: usize, n: usize,
           u: &mut [f32], s: &mut [f32], vt: &mut [f32]) -> SolverResult<()>;

    /// Cholesky: A = LL^T for positive definite A
    fn cholesky(&self, a: &mut [f32], n: usize) -> SolverResult<()>;
}
```

**GPU Strategy**: Block hybrid approach
1. **Panel factorization**: Small matrix operations on CPU (latency-sensitive)
2. **Trailing matrix update**: GEMM on GPU (compute-intensive)
3. **Look-ahead**: Overlap next panel with current update

For SVD, use one-sided Jacobi (arXiv:1707.05141) which is inherently parallel -- each rotation is independent. For large matrices, the divide-and-conquer approach (arXiv:2508.11467) achieves 1293x over rocSOLVER.

### 4D. World-Class Algorithmic Foundation

**LU**: Block LU with partial pivoting (Golub & Van Loan Ch. 3)
- Panel factorization + Schur complement update
- GPU: trailing update is GEMM (90%+ of compute)

**QR**: Householder with block reflectors (arXiv:1912.05508)
- TensorCore-accelerated: 2.9-14.7x speedup for tall-skinny matrices
- Block size tuned to shared memory capacity

**SVD**: Hierarchically blocked one-sided Jacobi (arXiv:1707.05141)
- Inherently parallel: each Jacobi rotation independent
- Maps to GPU memory hierarchy: warp-level → block-level → grid-level
- For large matrices: bidiagonal divide-and-conquer (arXiv:2508.11467)
- Randomized SVD for approximate low-rank (arXiv:2110.03423)

**Cholesky**: Block Cholesky with GEMM-heavy updates
- Positive definite only; ~2x faster than LU

**Reference Papers**:
1. Golub & Van Loan, "Matrix Computations", 4th ed., 2013
2. Efficient Batch SVD on GPUs, arXiv:2601.17979 (January 2026)
3. GPU Divide-and-Conquer SVD, arXiv:2508.11467 (1293x over rocSOLVER)
4. High Accuracy Low Precision QR, arXiv:1912.05508 (TensorCore QR)
5. Batched QR and SVD on GPUs, arXiv:1707.05141 (one-sided Jacobi)
6. Performant Unified GPU SVD Kernels, arXiv:2508.06339 (Julia-based)

---

## 5. trueno-image: GPU Image Processing

### 5A. Provable-Contracts Design

**Contract**: `image-conv2d-v1.yaml`

```yaml
metadata:
  version: "1.0.0"
  author: "Sovereign AI Stack"
  description: "2D image convolution (separable and non-separable)"
  references:
    - "Gonzalez & Woods, 'Digital Image Processing', 4th ed."
    - "Parallel Canny Edge Detection, PLOS One 2024"

equations:
  conv2d:
    formula: "O(i,j) = Σ_p Σ_q I(i+p, j+q) · K(p, q)"
    domain: "I ∈ ℝ^{H×W}, K ∈ ℝ^{kh×kw}"
    codomain: "O ∈ ℝ^{H×W} (same-padding)"
    invariants:
      - "Output dimensions match input (with same-padding)"
      - "Separable kernel: K = v · h^T gives same result as 2D convolution"
      - "Commutativity: I * K = K * I"

  separable_conv2d:
    formula: "O = (I * v) * h^T where K = v · h^T"
    domain: "I ∈ ℝ^{H×W}, v ∈ ℝ^{kh}, h ∈ ℝ^{kw}"
    codomain: "O ∈ ℝ^{H×W}"
    invariants:
      - "Result matches non-separable convolution within tolerance"
      - "Complexity: O(H·W·(kh+kw)) vs O(H·W·kh·kw)"

proof_obligations:
  - type: equivalence
    property: "Separable matches non-separable"
    formal: "|conv2d(I, v*h^T) - separable_conv2d(I, v, h)| < ε"
    tolerance: 1.0e-5
    applies_to: all

  - type: invariant
    property: "Gaussian blur is symmetric"
    formal: "blur(I, σ) has same result for any axis ordering"
    applies_to: all

  - type: equivalence
    property: "GPU matches CPU"
    tolerance: 4.0
    applies_to: simd

falsification_tests:
  - id: FALSIFY-IMG-001
    rule: "Separable equivalence"
    prediction: "Separable Gaussian matches 2D Gaussian within tolerance"
    test: "proptest with random images and Gaussian kernels"
    if_fails: "Separable decomposition or border handling error"

  - id: FALSIFY-IMG-002
    rule: "Identity kernel"
    prediction: "Convolution with delta kernel reproduces input exactly"
    test: "conv2d(I, [[0,0,0],[0,1,0],[0,0,0]]) == I"
    if_fails: "Off-by-one in kernel centering"

  - id: FALSIFY-IMG-003
    rule: "Canny edge detection"
    prediction: "Edges are 1-pixel wide (non-maximum suppression property)"
    test: "Verify no two adjacent pixels are both marked as edges along gradient direction"
    if_fails: "NMS implementation error"
```

### 5B. Brick Profiling Design

```rust
pub enum BrickId {
    // ... existing ...
    Conv2D,
    GaussianBlur,
    SobelFilter,
    Canny,
    Morphology,
    Histogram,
    Resize,
    ColorConvert,
}

pub enum BrickCategory {
    // ... existing ...
    ImageProcessing,
}
```

### 5C. Pure Rust Implementation

```rust
pub trait ImageOps {
    // Filtering
    fn conv2d(&self, image: &ImageBuf, kernel: &[f32], kw: usize, kh: usize,
              border: BorderMode) -> ImageBuf;
    fn gaussian_blur(&self, image: &ImageBuf, sigma: f32) -> ImageBuf;
    fn sobel(&self, image: &ImageBuf) -> (ImageBuf, ImageBuf); // (gx, gy)
    fn canny(&self, image: &ImageBuf, low: f32, high: f32) -> ImageBuf;

    // Morphology
    fn dilate(&self, image: &ImageBuf, element: &StructElement) -> ImageBuf;
    fn erode(&self, image: &ImageBuf, element: &StructElement) -> ImageBuf;

    // Geometry
    fn resize(&self, image: &ImageBuf, new_w: usize, new_h: usize,
              interp: Interpolation) -> ImageBuf;

    // Color
    fn rgb_to_gray(&self, image: &ImageBuf) -> ImageBuf;
    fn rgb_to_hsv(&self, image: &ImageBuf) -> ImageBuf;

    // Analysis
    fn histogram(&self, image: &ImageBuf, bins: usize) -> Vec<u32>;
    fn connected_components(&self, binary: &ImageBuf) -> (ImageBuf, u32);
}

pub struct ImageBuf {
    pub data: Vec<u8>,     // or Vec<f32>
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub dtype: ImageDType,
}

pub enum BorderMode { Replicate, Reflect, Zero, Wrap }
pub enum Interpolation { Nearest, Bilinear, Bicubic, Lanczos }
```

**GPU Strategy**: Each operation is a separate PTX kernel:
- **Conv2D**: Shared memory tiling with halo (border elements loaded from global)
- **Gaussian blur**: Two 1D passes (separable) for O(kh+kw) instead of O(kh*kw)
- **Sobel**: 3x3 shared memory tile, 8 neighbor reads per pixel
- **Histogram**: Per-block shared memory histogram + global atomicAdd reduction
- **Resize**: Output-centric parallelism (each thread computes one output pixel)

### 5D. World-Class Algorithmic Foundation

**Canny Pipeline** (GPU, per PLOS One 2024):
1. Gaussian blur (separable, shared memory)
2. Sobel gradient magnitude + direction
3. Non-maximum suppression (per-pixel, gradient direction lookup)
4. Hysteresis thresholding (parallel connected component propagation)

**Separable Convolution**: Two 1D passes — first horizontal (row-parallel), then vertical (column-parallel). Each pass uses shared memory tiling with halo regions. This is the standard approach used by NPP internally.

**Reference Papers**:
1. Parallel Canny Edge Detection, PLOS One 2024 (1.21x over CUDA Canny)
2. cubic: CUDA-accelerated Bioimage Computing, arXiv:2510.14143

---

## 6. trueno-rand: Parallel Random Number Generation

### 6A. Provable-Contracts Design

**Contract**: `rand-philox-v1.yaml`

```yaml
metadata:
  version: "1.0.0"
  author: "Sovereign AI Stack"
  description: "Philox 4x32-10 counter-based PRNG for GPU"
  references:
    - "Salmon et al., 'Parallel Random Numbers: As Easy as 1, 2, 3', SC 2011"

equations:
  philox4x32:
    formula: "output = philox_round^10(counter, key) where each round applies Feistel network"
    domain: "counter ∈ ℤ^4 (128-bit), key ∈ ℤ^2 (64-bit)"
    codomain: "output ∈ {0,1}^128 (4 × 32-bit random values)"
    invariants:
      - "Deterministic: same (counter, key) always produces same output"
      - "Period: 2^128 (counter space)"
      - "Statistical quality: passes BigCrush (TestU01)"

proof_obligations:
  - type: invariant
    property: "Determinism"
    formal: "philox(c, k) == philox(c, k) always"
    applies_to: all

  - type: invariant
    property: "Counter independence"
    formal: "philox(c1, k) and philox(c2, k) are statistically independent for c1 ≠ c2"
    applies_to: all

  - type: equivalence
    property: "GPU matches CPU bit-exact"
    tolerance: 0.0
    applies_to: simd

falsification_tests:
  - id: FALSIFY-RNG-001
    rule: "Determinism"
    prediction: "Same counter+key always produces same output"
    test: "Generate twice with same state, compare bit-exact"
    if_fails: "Uninitialized state or race condition"

  - id: FALSIFY-RNG-002
    rule: "Uniformity"
    prediction: "Chi-squared test passes for bucket distribution"
    test: "Generate 1M samples, chi-squared goodness-of-fit"
    if_fails: "Round function is biased"

  - id: FALSIFY-RNG-003
    rule: "GPU-CPU bit-exactness"
    prediction: "GPU and CPU produce identical bits"
    test: "Generate 10K values on GPU and CPU, compare bit-exact"
    if_fails: "Integer arithmetic divergence (should be impossible for Philox)"

kani_harnesses:
  - id: KANI-RNG-001
    obligation: "Feistel round invertibility"
    property: "Each round is bijective (no information loss)"
    bound: 4
    strategy: exhaustive
    harness: verify_feistel_bijective
```

### 6B-C. Implementation

**Philox 4x32-10** (Salmon et al., SC 2011):
- Counter-based: embarrassingly parallel (no state sharing between threads)
- Each GPU thread computes `philox(global_thread_id + offset, key)` for 4 random u32s
- 10 rounds of Feistel network using widening multiply
- Passes BigCrush statistical test suite

```rust
pub trait GpuRng {
    fn fill_uniform_f32(&self, output: &mut GpuBuffer<f32>, seed: u64, offset: u64);
    fn fill_normal_f32(&self, output: &mut GpuBuffer<f32>, seed: u64, mean: f32, std: f32);
    fn fill_uniform_u32(&self, output: &mut GpuBuffer<u32>, seed: u64, offset: u64);
}
```

**Normal distribution**: Box-Muller transform applied to pairs of uniform values.

### 6D. World-Class Algorithmic Foundation

**Reference**: Salmon et al., "Parallel Random Numbers: As Easy as 1, 2, 3", SC 2011
- Counter-based design: no per-thread state, perfect for GPU
- Philox: 10 rounds of Feistel using 32×32→64 widening multiply
- Threefry: 20 rounds using rotations and XOR (backup for non-multiply hardware)
- Both pass BigCrush and Crush from TestU01

---

## 7. trueno-tensor: N-Dimensional Tensor Contractions

### 7A. Provable-Contracts Design

**Contract**: `tensor-contraction-v1.yaml`

```yaml
metadata:
  version: "1.0.0"
  description: "General tensor contraction (Einstein summation)"
  references:
    - "Springer et al., 'Design of a High-Performance GEMM-like Tensor-Tensor Multiplication', ACM TOMS 2018"

equations:
  contraction:
    formula: "C_{free_C} = α · Σ_{contracted} A_{free_A, contracted} · B_{free_B, contracted} + β · C_{free_C}"
    domain: "A ∈ ℝ^{d1×...×dk}, B ∈ ℝ^{dk+1×...×dn}"
    codomain: "C with dimensions = free indices of A ∪ free indices of B"
    invariants:
      - "Contracted dimensions must match between A and B"
      - "Output dimensions = free_A dimensions × free_B dimensions"
      - "For rank-2 tensors, reduces to GEMM"

proof_obligations:
  - type: invariant
    property: "Reduces to GEMM for matrices"
    formal: "contraction(A[m,k], B[k,n], 'ik,kj->ij') == gemm(A, B)"
    tolerance: 1.0e-5
    applies_to: all

  - type: invariant
    property: "Output shape correctness"
    formal: "output_shape == infer_shape(A_shape, B_shape, contraction_indices)"
    applies_to: all

  - type: associativity
    property: "Contraction order independence"
    formal: "|contract(contract(A,B), C) - contract(A, contract(B,C))| < ε"
    tolerance: 1.0e-3
    applies_to: all
```

### 7C. Pure Rust Implementation

**Strategy**: Reduce all tensor contractions to GEMM via TTGT (Transpose-Transpose-GEMM-Transpose):
1. Reshape/transpose A to 2D: (free_A, contracted)
2. Reshape/transpose B to 2D: (contracted, free_B)
3. Call existing trueno GEMM
4. Reshape/transpose result to output dimensions

This leverages the heavily optimized GEMM already in trueno-gpu (tiled, tensor core variants).

For specialized contractions (e.g., batched outer products, trace), emit custom PTX kernels that avoid the transpose overhead.

```rust
pub fn einsum(
    subscripts: &str,          // e.g., "ijk,jkl->il"
    inputs: &[&Tensor],
    output: &mut Tensor,
) -> Result<(), TensorError>;
```

---

## 8. Enhanced Dense BLAS (cuBLAS Parity)

### Missing Operations (Priority Order)

| Operation | cuBLAS Name | Status | Priority |
|---|---|---|---|
| General GEMM | gemm | Exists (tiled, tensor core) | Done |
| Batched GEMM | gemmBatched | Exists (batched_4d) | Done |
| Strided Batched | gemmStridedBatched | Partial | P1 |
| Mixed Precision | gemmEx | Missing | P1 |
| Symmetric Rank-k | syrk/syr2k | Missing | P2 |
| Triangular Solve | trsm | Missing (needed for solvers) | P1 |
| Triangular Multiply | trmm | Missing | P2 |
| Symmetric Multiply | symm/hemm | Missing | P2 |
| Level-2 GEMV | gemv | Exists (CoalescedGemvKernel) | Done |
| Level-1 (dot, nrm2, etc.) | Various | Exists in trueno core | Done |
| Epilogue Fusion | gemmEx with epilogue | Missing | P2 |

**Contract**: `blas-trsm-v1.yaml` — triangular solve is critical for LU/QR/Cholesky solvers.

```yaml
equations:
  trsm:
    formula: "Solve AX = B for X, where A is triangular"
    domain: "A ∈ ℝ^{n×n} (triangular), B ∈ ℝ^{n×nrhs}"
    codomain: "X ∈ ℝ^{n×nrhs}"
    invariants:
      - "AX = B within backward error"
      - "||AX - B||_F / (||A||_F · ||X||_F) < n · u"
```

---

## 9. Enhanced Compression (nvCOMP Parity)

### Current State

trueno has LZ4 CPU + GPU kernels. trueno-zram provides ZRAM integration. Missing:

| Algorithm | nvCOMP | Sovereign Stack | Priority |
|---|---|---|---|
| LZ4 | Yes | Yes (trueno LZ4 kernel) | Done |
| Zstd | Yes | trueno-zram (CPU) | P1: GPU kernel |
| Snappy | Yes | Missing | P3 |
| GDeflate | Yes | Missing | P3 |
| Cascaded | Yes | Missing | P3 |
| ANS | Yes | Missing | P2 |

**Strategy**: Focus on Zstd GPU kernel (most impactful for KV cache compression in realizar) and ANS (used for neural codec compression).

---

## 10. Implementation Roadmap

### Phase 1: Foundation (Months 1-3)

| Priority | Crate | Deliverable | Contract |
|---|---|---|---|
| P0 | trueno-sparse | CSR/COO formats + CPU SIMD SpMV | sparse-spmv-v1.yaml |
| P0 | trueno-sparse | Merge-based GPU SpMV kernel | sparse-spmv-v1.yaml |
| P0 | trueno-gpu | Triangular solve (TRSM) kernel | blas-trsm-v1.yaml |
| P0 | trueno-gpu | Mixed-precision GEMM (FP16 in, FP32 acc) | blas-gemmex-v1.yaml |

### Phase 2: Solvers + FFT (Months 3-6)

| Priority | Crate | Deliverable | Contract |
|---|---|---|---|
| P1 | trueno-solve | LU factorization (block, GPU) | solve-lu-v1.yaml |
| P1 | trueno-solve | QR factorization (Householder, GPU) | solve-qr-v1.yaml |
| P1 | trueno-solve | SVD (one-sided Jacobi, batched) | solve-svd-v1.yaml |
| P1 | trueno-fft | 1D C2C + R2C Stockham FFT (GPU) | fft-stockham-v1.yaml |
| P1 | trueno-fft | 2D FFT via 1D + transpose | fft-2d-v1.yaml |
| P1 | trueno-sparse | SpMM kernel (row-split + merge) | sparse-spmm-v1.yaml |

### Phase 3: Image + Ecosystem (Months 6-9)

| Priority | Crate | Deliverable | Contract |
|---|---|---|---|
| P2 | trueno-image | Separable conv2d + Gaussian blur | image-conv2d-v1.yaml |
| P2 | trueno-image | Sobel + Canny edge detection | image-canny-v1.yaml |
| P2 | trueno-image | Histogram + morphology | image-histogram-v1.yaml |
| P2 | trueno-rand | Philox 4x32-10 GPU RNG | rand-philox-v1.yaml |
| P2 | trueno-tensor | Einstein summation via TTGT | tensor-contraction-v1.yaml |
| P2 | trueno-solve | Cholesky factorization | solve-cholesky-v1.yaml |

### Phase 4: Polish + Parity (Months 9-12)

| Priority | Crate | Deliverable | Contract |
|---|---|---|---|
| P3 | trueno-sparse | SpGEMM (sparse × sparse) | sparse-spgemm-v1.yaml |
| P3 | trueno-sparse | BSR + SELL formats | sparse-formats-v1.yaml |
| P3 | trueno-fft | 3D FFT + batched | fft-3d-v1.yaml |
| P3 | trueno-fft | Bluestein non-power-of-two | fft-bluestein-v1.yaml |
| P3 | trueno-image | Resize (bilinear/bicubic/Lanczos) | image-resize-v1.yaml |
| P3 | trueno-image | Color conversion + connected components | image-color-v1.yaml |
| P3 | trueno-gpu | Zstd GPU compression kernel | comp-zstd-v1.yaml |

---

## 11. Quality Gates (Per PAIML MCP Agent Toolkit Standards)

Every new crate and kernel must pass before merge to master:

### Code Quality

| Gate | Threshold | Tool |
|---|---|---|
| Test Coverage | >= 95% line | `cargo llvm-cov` |
| Mutation Score | >= 80% kill | `cargo mutants` |
| TDG Grade | >= A- (88+) | `pmat tdg` |
| Cyclomatic Complexity | <= 15 per fn | `pmat complexity` |
| SATD Count | 0 | `pmat satd` |
| Clippy | 0 warnings | `cargo clippy -- -D warnings` |
| Rustfmt | Pass | `cargo fmt --check` |

### Numerical Quality

| Gate | Verification | Tool |
|---|---|---|
| Backward Error | Per-contract tolerance | proptest + contract tests |
| SIMD-Scalar Parity | <= 8 ULP | proptest with ULP comparison |
| GPU-CPU Parity | <= 32 ULP | contract tests with GPU runner |
| Known-Value Tests | Exact match | Hardcoded reference values |
| Property Tests | 10K+ iterations | proptest |
| Kani Verification | All harnesses pass | `cargo kani` |

### Performance Quality

| Gate | Threshold | Tool |
|---|---|---|
| Roofline Efficiency | >= 50% of theoretical | BrickProfiler + roofline model |
| Budget Compliance | Meet contract budget | ComputeBrick.verify() |
| No Performance Regression | < 5% degradation | Criterion.rs + CI baseline |

### Provable-Contracts Pipeline

For each new operation:
1. **Phase 1 (Extract)**: Identify paper, extract equations
2. **Phase 2 (Specify)**: Write YAML contract with obligations + falsification tests
3. **Phase 3 (Scaffold)**: Generate trait + test stubs via `pv scaffold`
4. **Phase 4 (Implement)**: Scalar reference first, then SIMD, then GPU
5. **Phase 5 (Falsify)**: Property-based tests via `pv probar`
6. **Phase 6 (Verify)**: Kani bounded model checking via `pv verify`
7. **Phase 7 (Prove)**: Lean 4 proofs for critical properties (optional)

---

## 12. Integration with Sovereign Stack Components

### Batuta Oracle Integration Points

Per oracle recommendations:
- **trueno** is the compute foundation (85-95% confidence for linear algebra)
- **provable-contracts** validates correctness (95% confidence for verification)
- **pepita** provides kernel-level I/O primitives (io_uring, ublk)
- **repartir** distributes multi-GPU work (CPU/GPU executors, checkpointing)

### Downstream Consumer Benefits

| Consumer | New Capability Unlocked |
|---|---|
| **aprender** | PCA via SVD, spectral clustering via eigendecomp, graph Laplacian via sparse |
| **entrenar** | FFT-based convolution layers, sparse attention, QR-based optimizer |
| **realizar** | Sparse attention patterns, FFT for audio models, image preprocessing |
| **trueno-rag** | Sparse retrieval indices, FFT for audio chunking |
| **rmedia** | GPU image processing pipeline (resize, filter, color convert) |
| **simular** | FFT for physics simulation, sparse systems for FEM |

### Cross-Crate Contract Binding

Each new trueno capability gets a binding entry in `provable-contracts/contracts/trueno/binding.yaml`:

```yaml
version: "1.0.0"
target_crate: trueno
bindings:
  - contract: sparse-spmv-v1.yaml
    equation: spmv
    module_path: "trueno_sparse::ops::spmv_csr"
    function: spmv_csr
    status: not_implemented

  - contract: fft-stockham-v1.yaml
    equation: dft
    module_path: "trueno_fft::fft_1d"
    function: fft_1d
    status: not_implemented
```

---

## 13. Competitive Positioning

### Sovereign Stack Advantages Over CUDA Libraries

| Advantage | Details |
|---|---|
| **No vendor lock-in** | wgpu backend supports Vulkan, Metal, DX12, WebGPU |
| **Pure Rust safety** | Zero unsafe in public API, ownership-based memory safety |
| **Zero external toolchains** | No nvcc, no LLVM, no C/C++ compiler needed |
| **Cross-platform** | Same code runs on CPU SIMD, GPU, and WASM |
| **Formal verification** | provable-contracts with Kani + Lean 4 (CUDA has no equivalent) |
| **ML-based auto-tuning** | BrickTuner with learned kernel selection (CUDA relies on manual selection) |
| **Built-in profiling** | Every operation profiled via Brick with roofline analysis |
| **Cargo ecosystem** | `cargo add trueno-sparse` vs manual CMake + library linking |

### Where CUDA Libraries Still Lead

| Gap | CUDA Advantage | Timeline to Close |
|---|---|---|
| **Maturity** | 15+ years of optimization | 2-3 years for parity |
| **Sparse algorithm breadth** | 20+ SpMV algorithms, preconditioners | 6-12 months for core |
| **FFT performance** | Heavily tuned for each GPU architecture | 6-12 months for competitive |
| **Solver completeness** | Full LAPACK equivalent | 12-18 months |
| **Image processing breadth** | 200+ NPP functions | 12-24 months for core 50 |
| **Multi-GPU** | NCCL, cuBLASMp | Partially covered by repartir |

---

## Appendix A: Batuta Oracle Capability Map

```
┌──────────────────────────────────────────────────────────────┐
│                    Sovereign AI Stack                         │
│                                                              │
│  ┌─────────────────── Orchestration ──────────────────────┐  │
│  │ batuta (workflow)  repartir (distributed)  pforge (MCP) │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─────────────── Training & Inference ───────────────────┐  │
│  │ entrenar (train)   realizar (infer)   whisper-apr (ASR) │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────── ML Algorithms ─────────────────────┐  │
│  │ aprender (classical ML + neural + text + audio)         │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─────────────── Compute Primitives ─────────────────────┐  │
│  │ trueno (SIMD/GPU)   trueno-gpu (PTX)   pepita (kernel)  │  │
│  │ trueno-db (analytics)  trueno-graph  trueno-rag          │  │
│  │                                                          │  │
│  │ ┌── NEW (This Spec) ─────────────────────────────────┐  │  │
│  │ │ trueno-sparse    trueno-fft     trueno-solve        │  │  │
│  │ │ trueno-image     trueno-rand    trueno-tensor       │  │  │
│  │ └────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────── Quality & Profiling ───────────────────┐  │
│  │ pmat (TDG)  certeza (coverage)  provable-contracts      │  │
│  │ probar (property)  renacer (strace)  batuta (oracle)    │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌─────────────────── Transpilers ────────────────────────┐  │
│  │ depyler (Py→Rust)  decy (C→Rust)  ruchy (script→Rust)  │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

## Appendix B: CUDA Library → Sovereign Stack Mapping

| CUDA Library | Sovereign Crate | Status |
|---|---|---|
| cuBLAS Level 1 | trueno (vector_ops) | Complete |
| cuBLAS Level 2 | trueno (GEMV) + trueno-gpu (CoalescedGemvKernel) | ~80% |
| cuBLAS Level 3 | trueno-gpu (GEMM variants, batched) | ~70% |
| cuBLASLt | trueno-gpu (algorithm selection via BrickTuner) | ~50% |
| cuBLASMp | repartir (distributed executor) | ~30% |
| cuSPARSE | **trueno-sparse** (NEW) | 0% → Phase 1 |
| cuSPARSELt | **trueno-sparse** (structured sparsity) | 0% → Phase 3 |
| cuFFT | **trueno-fft** (NEW) | 0% → Phase 2 |
| cuFFTMp | repartir + trueno-fft (distributed) | 0% → Phase 4 |
| cuSOLVER Dense | **trueno-solve** (NEW) | 5% → Phase 2 |
| cuSOLVER Sparse | trueno-solve + trueno-sparse | 0% → Phase 3 |
| cuTENSOR | **trueno-tensor** (NEW) | 0% → Phase 3 |
| NPP | **trueno-image** (NEW) | 0% → Phase 3 |
| nvCOMP | trueno (LZ4) + trueno-zram | ~20% → Phase 4 |
| nvJPEG | **trueno-image** (codec module) | 0% → Phase 4 |
| cuRAND | **trueno-rand** (NEW) | 0% → Phase 3 |
| cuPQC | aprender (CPU crypto) | ~10% → Future |
| MathDx | trueno-gpu (device-side math) | ~40% |

---

*This specification is a living document. Each phase completion triggers a review and update cycle.*

*Generated with guidance from batuta oracle, provable-contracts framework, and arXiv research survey.*

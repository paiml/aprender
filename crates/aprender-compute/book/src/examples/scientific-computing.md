# Scientific Computing

Trueno provides cuSOLVER, cuSPARSE, cuTENSOR, and cuRAND parity for scientific workflows.

## Dense Solvers (trueno-solve)

### LU Factorization

```rust
use trueno_solve::lu_factorize;

let lu = lu_factorize(&matrix, n)?;
let x = lu.solve(&rhs)?;
```

### QR Factorization (Least Squares)

```rust
use trueno_solve::qr_factorize;

let qr = qr_factorize(&matrix, m, n)?;  // m >= n
let x = qr.solve(&rhs)?;
```

### SVD

```rust
use trueno_solve::svd;

let result = svd(&matrix, m, n)?;
// result.sigma: singular values (decreasing)
// result.u, result.vt: left/right singular vectors
```

### Cholesky (Symmetric Positive Definite)

```rust
use trueno_solve::cholesky;

let chol = cholesky(&spd_matrix, n)?;
let x = chol.solve(&rhs)?;
```

### TRSM (Triangular Solve)

```rust
use trueno_solve::{trsm, TriangularSide, DiagonalType};

let result = trsm(&tri, &rhs, n, nrhs, TriangularSide::Lower, DiagonalType::NonUnit)?;
```

### BLAS Level-3

```rust
use trueno_solve::{syrk, syr2k, trmm, symm};

syrk(&a, &mut c, n, k, alpha, beta)?;      // C = α·A·Aᵀ + β·C
syr2k(&a, &b, &mut c, n, k, alpha, beta)?; // C = α·(A·Bᵀ + B·Aᵀ) + β·C
trmm(&a, &mut b, n, nrhs, alpha)?;          // B = α·A·B (triangular)
symm(&a, &b, &mut c, n, m, alpha, beta)?;   // C = α·A·B + β·C (symmetric)
```

### Mixed-Precision GEMM (gemmEx)

```rust
use trueno_solve::{gemm_ex, f32_to_f16};

// f16 inputs, f32 accumulation (cuBLAS gemmEx parity)
let a: Vec<u16> = floats_a.iter().map(|&v| f32_to_f16(v)).collect();
let b: Vec<u16> = floats_b.iter().map(|&v| f32_to_f16(v)).collect();
gemm_ex(&a, &b, &mut c, m, n, k, alpha, beta)?;
```

### Strided Batched GEMM

```rust
use trueno_solve::gemm_strided_batched;

// Batch of matmuls: C_b = α·A_b·B_b + β·C_b
gemm_strided_batched(
    &a, stride_a, &b, stride_b, &mut c, stride_c,
    batch_count, m, n, k, alpha, beta,
)?;
```

## Sparse Algebra (trueno-sparse)

### Formats

| Format | Best For | API |
|--------|----------|-----|
| CSR | General sparse | `CsrMatrix::new(...)` |
| COO | Construction | `CooMatrix::new(...)` |
| BSR | Block-structured | `BsrMatrix::from_dense(...)` |
| SELL | SIMD-friendly | `SellMatrix::from_csr(...)` |

### SpMV / SpMM

```rust
use trueno_sparse::{CsrMatrix, SparseOps};

csr.spmv(alpha, &x, beta, &mut y)?;           // y = α·A·x + β·y
csr.spmm(alpha, &b, b_cols, beta, &mut c)?;   // C = α·A·B + β·C
```

### SpGEMM (Sparse × Sparse)

```rust
use trueno_sparse::spgemm;

let c = spgemm(&a, &b)?;  // CSR × CSR → CSR (Gustavson's algorithm)
```

## Tensor Contractions (trueno-tensor)

Einstein summation via TTGT (Transpose-Transpose-GEMM-Transpose):

```rust
use trueno_tensor::{einsum, einsum_nary, matmul, outer, trace, Tensor};

let c = matmul(&a, &b)?;                   // Matrix multiply
let c = einsum("ijk,jkl->il", &t1, &t2)?;  // Arbitrary contraction
let op = outer(&u, &v)?;                    // Outer product
let tr = trace(&matrix)?;                   // Trace

// N-ary einsum: chain of contractions
let chain = einsum_nary("ij,jk,kl->il", &[&a, &b, &c])?;
```

## Random Number Generation (trueno-rand)

Two counter-based PRNGs with cuRAND parity:

### Philox 4×32-10 (multiply-based)

```rust
use trueno_rand::Philox4x32;

let mut rng = Philox4x32::new(seed);
rng.fill_uniform(&mut buffer);   // U[0, 1)
rng.fill_normal(&mut buffer);    // N(0, 1)

// Stateless generation (GPU-friendly)
let vals = Philox4x32::generate_at(key, counter);
```

### Threefry 4×64-20 (rotation-based, no multiply)

```rust
use trueno_rand::Threefry4x64;

let mut rng = Threefry4x64::new(seed);
rng.fill_uniform(&mut buffer);   // U[0, 1)
rng.fill_normal(&mut buffer);    // N(0, 1)

// Stateless generation
let vals = Threefry4x64::generate_at(key, counter);
```

## Provable Contracts

Every operation has a YAML contract with proof obligations:
- **Backward error bounds** per LAProof
- **Roundtrip accuracy** (e.g., IFFT(FFT(x)) = x)
- **Property tests** via proptest (10K+ iterations)
- **Known-value verification** against reference implementations

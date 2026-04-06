# Provable Contracts Pipeline

Trueno uses YAML-defined provable contracts to enforce correctness invariants at compile time. Contracts are checked as `debug_assert!` in debug builds and compiled away in release builds.

## Contract Files

All contracts live in `contracts/`:

| Contract | Domain |
|----------|--------|
| `gemv-kernel-v1.yaml` | General matrix-vector multiply |
| `elementwise-kernel-v1.yaml` | Element-wise operations |
| `softmax-kernel-v1.yaml` | Softmax numerical stability |
| `transpose-kernel-v1.yaml` | Transpose correctness |
| `blas-level3-v1.yaml` | BLAS Level 3 (GEMM, TRSM) |
| `blas-trsm-v1.yaml` | Triangular solve |
| `tensor-contraction-v1.yaml` | Tensor contraction |
| `sparse-spmv-v1.yaml` | Sparse matrix-vector multiply |
| `sparse-spgemm-v1.yaml` | Sparse matrix-matrix multiply |
| `sparse-spmm-v1.yaml` | Sparse-dense multiply |
| `sparse-formats-v1.yaml` | Sparse format invariants |
| `sparse-bsr-v1.yaml` | Block sparse row |
| `fft-stockham-v1.yaml` | FFT Stockham algorithm |
| `fft-bluestein-v1.yaml` | FFT Bluestein algorithm |
| `fft-2d-v1.yaml` | 2D FFT |
| `fft-3d-v1.yaml` | 3D FFT |
| `solve-lu-v1.yaml` | LU decomposition solver |
| `solve-qr-v1.yaml` | QR decomposition solver |
| `solve-cholesky-v1.yaml` | Cholesky decomposition solver |
| `solve-svd-v1.yaml` | SVD decomposition solver |
| `image-conv2d-v1.yaml` | 2D convolution |
| `image-resize-v1.yaml` | Image resizing |
| `image-canny-v1.yaml` | Canny edge detection |
| `image-color-v1.yaml` | Color space conversion |
| `image-histogram-v1.yaml` | Histogram computation |
| `rand-philox-v1.yaml` | Philox PRNG |
| `rand-threefry-v1.yaml` | ThreeFry PRNG |

## How the Pipeline Works

1. **YAML** -- Each contract defines preconditions, postconditions, and invariants
2. **build.rs** -- At compile time, `build.rs` reads the YAML files and generates Rust assertion code
3. **#[contract]** -- Functions annotated with `#[contract]` get the generated checks injected
4. **debug_assert!** -- Checks run in debug/test builds; zero overhead in release

## Running the Demo

```bash
cargo run --example contract_pipeline_demo
```

## Adding a New Contract

1. Create `contracts/my-feature-v1.yaml` with preconditions and postconditions
2. Add precondition checks to the target function
3. Annotate the function with `#[contract]`
4. Run tests: `cargo test`

For the full contract specification and YAML schema, see the
[provable-contracts integration guide](https://github.com/paiml/provable-contracts/blob/main/book/src/integration.md).

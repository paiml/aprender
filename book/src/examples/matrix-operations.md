# Matrix Operations

This chapter demonstrates Trueno's matrix operations using the `matrix_operations` example.

## Running the Example

```bash
cargo run --example matrix_operations
```

## Basic Matrix Operations

### Creating Matrices

```rust
use trueno::Matrix;

// Create from row-major data
let a = Matrix::from_vec(2, 3, vec![
    1.0, 2.0, 3.0,  // Row 0
    4.0, 5.0, 6.0,  // Row 1
])?;

// Identity matrix
let identity = Matrix::identity(3);

// Zero matrix
let zeros = Matrix::zeros(2, 3);
```

### Matrix Multiplication

```rust
// C = A × B
let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
let b = Matrix::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])?;

let c = a.matmul(&b)?;  // Result: 2×2 matrix
```

### Matrix-Vector Multiplication

```rust
use trueno::{Matrix, Vector};

let weights = Matrix::from_vec(3, 4, weight_data)?;
let input = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);

let output = weights.matvec(&input)?;  // Result: Vector of length 3
```

### Transpose

```rust
let a = Matrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
let at = a.transpose();  // Result: 3×2 matrix
```

## Neural Network Layers

### Linear Layer (Dense)

```rust
fn linear_layer(
    input: &Vector,
    weights: &Matrix,
    bias: &Vector,
) -> Result<Vector, TruenoError> {
    let output = weights.matvec(input)?;
    output.add(bias)
}
```

### Batch Processing

```rust
// Process multiple samples through the same layer
let samples = vec![
    Vector::from_slice(&[0.2, -0.3, 0.5]),
    Vector::from_slice(&[0.3, 0.0, 0.1]),
    Vector::from_slice(&[0.0, 0.3, 0.4]),
];

for sample in &samples {
    let output = weights.matvec(sample)?;
    println!("Output: {:?}", output.as_slice());
}
```

## Mathematical Properties

The example verifies key mathematical properties:

### Identity Property

```rust
// I × v = v
let identity = Matrix::identity(3);
let v = Vector::from_slice(&[1.0, 2.0, 3.0]);
let result = identity.matvec(&v)?;
assert_eq!(result.as_slice(), v.as_slice());
```

### Transpose Property

```rust
// (A × v)^T = v^T × A^T
// This is verified in the example
```

### Zero Property

```rust
// A × 0 = 0
let zeros = Vector::from_slice(&[0.0, 0.0, 0.0, 0.0]);
let result = weights.matvec(&zeros)?;
// All elements should be 0
```

## Batched Matrix Multiplication

### 3D Tensors (Batch Processing)

Process multiple matrix multiplications in a single call:

```rust
use trueno::Matrix;

// Shape: [batch, m, k] @ [batch, k, n] -> [batch, m, n]
let batch = 4;
let m = 32;
let k = 64;
let n = 32;

let a_data: Vec<f32> = vec![0.1; batch * m * k];
let b_data: Vec<f32> = vec![0.2; batch * k * n];

let result = Matrix::batched_matmul(&a_data, &b_data, batch, m, k, n)?;
```

### 4D Tensors (Multi-Head Attention)

The exact pattern used in transformer attention (`Q @ K^T` and `attn @ V`):

```rust
// Simulate multi-head attention: Q @ K^T
// Shape: [batch, heads, seq, head_dim] @ [batch, heads, head_dim, seq]
let batch = 1;
let heads = 12;
let seq_len = 512;
let head_dim = 64;

let q_data: Vec<f32> = vec![0.0; batch * heads * seq_len * head_dim];
let kt_data: Vec<f32> = vec![0.0; batch * heads * head_dim * seq_len];

let attn_scores = Matrix::batched_matmul_4d(
    &q_data,
    &kt_data,
    batch,
    heads,
    seq_len,   // m
    head_dim,  // k
    seq_len,   // n
)?;
// Output: [batch, heads, seq_len, seq_len] attention scores
```

This is critical for transformer inference performance - each (batch, head) pair is processed independently using SIMD matmul, achieving ~50 GFLOPS vs ~0.1 GFLOPS for naive implementation.

## Performance Considerations

### Blocking for Cache Efficiency

Trueno uses blocked matrix multiplication for better cache utilization:

```rust
// Automatic blocking for large matrices
let large_a = Matrix::from_vec(1024, 1024, data_a)?;
let large_b = Matrix::from_vec(1024, 1024, data_b)?;
let c = large_a.matmul(&large_b)?;  // Uses tiled algorithm internally
```

### SIMD Acceleration

Matrix operations automatically use SIMD when beneficial:

- AVX2: Process 8 f32 values per instruction
- AVX-512: Process 16 f32 values per instruction
- Automatic fallback to scalar for small matrices

### GPU Acceleration

For large matrices, enable GPU acceleration:

```bash
cargo run --release --features gpu --example matrix_operations
```

The GPU backend is automatically selected for matrices above the threshold (typically 256×256).

## Benchmark Suite

Run the matrix benchmark suite:

```bash
cargo run --release --example benchmark_matrix_suite
```

This compares:
- Naive O(n³) multiplication
- SIMD-optimized blocked multiplication
- Parallel (rayon) multiplication

## See Also

- [Eigendecomposition](../api-reference/eigendecomposition.md) - Symmetric eigenvalue decomposition
- [GPU Backend](../architecture/gpu-backend.md) - GPU acceleration details
- [Performance Targets](../performance/targets.md) - Expected speedups

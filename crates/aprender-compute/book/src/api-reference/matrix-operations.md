# Matrix Operations

The `Matrix<T>` type provides 2D matrix operations with SIMD acceleration.

## Creating Matrices

```rust
use trueno::Matrix;

// From dimensions (uninitialized)
let m = Matrix::<f32>::new(3, 4);

// From Vec with dimensions
let m = Matrix::<f32>::from_vec(2, 3, vec![
    1.0, 2.0, 3.0,
    4.0, 5.0, 6.0,
])?;

// Special matrices
let zeros = Matrix::<f32>::zeros(3, 3);
let identity = Matrix::<f32>::identity(4);
```

## Basic Properties

```rust
let m = Matrix::<f32>::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;

m.rows();        // 2
m.cols();        // 3
m.len();         // 6 (total elements)
m.as_slice();    // &[f32] view of data
m.get(0, 1);     // Some(2.0)
m.get_mut(1, 2); // Mutable access
```

## Matrix Multiplication

```rust
let a = Matrix::<f32>::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
let b = Matrix::<f32>::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0])?;

// Matrix-matrix multiplication: [2×3] × [3×2] = [2×2]
let c = a.matmul(&b)?;
```

## Matrix-Vector Multiplication

```rust
use trueno::Vector;

let m = Matrix::<f32>::from_vec(3, 4, vec![/* 12 elements */])?;
let v = Vector::<f32>::from_slice(&[1.0, 2.0, 3.0, 4.0]);

// Matrix × Vector: [3×4] × [4×1] = [3×1]
let result = m.matvec(&v)?;

// Vector × Matrix: [1×3] × [3×4] = [1×4]
let v2 = Vector::<f32>::from_slice(&[1.0, 2.0, 3.0]);
let result = m.vecmat(&v2)?;
```

## Transpose

```rust
let m = Matrix::<f32>::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;

// [2×3] → [3×2]
let mt = m.transpose();
```

## Convolution (2D)

For image processing and CNNs.

```rust
let image = Matrix::<f32>::from_vec(5, 5, /* 25 elements */)?;
let kernel = Matrix::<f32>::from_vec(3, 3, vec![
    1.0, 0.0, -1.0,
    2.0, 0.0, -2.0,
    1.0, 0.0, -1.0,
])?; // Sobel edge detection

let edges = image.convolve2d(&kernel)?;
```

## Embedding Lookup

For NLP models (word embeddings).

```rust
// Embedding table: vocab_size × embedding_dim
let embeddings = Matrix::<f32>::from_vec(1000, 128, /* ... */)?;

// Token indices
let tokens: Vec<usize> = vec![42, 7, 256, 13];

// Lookup: returns [4×128] matrix
let token_embeddings = embeddings.embedding_lookup(&tokens)?;
```

## Batched Matrix Multiplication (3D Tensors)

For batch processing of independent matrix multiplications:

```rust
// Shape: [batch, m, k] @ [batch, k, n] -> [batch, m, n]
let batch = 4;
let m = 32;
let k = 64;
let n = 32;

// Flattened input tensors
let a_data: Vec<f32> = vec![0.0; batch * m * k];
let b_data: Vec<f32> = vec![0.0; batch * k * n];

let result = Matrix::batched_matmul(&a_data, &b_data, batch, m, k, n)?;
// Result: Vec<f32> with shape [batch, m, n]
```

## Batched 4D Matrix Multiplication (Attention Pattern)

For multi-head attention in transformers:

```rust
// Shape: [batch, heads, m, k] @ [batch, heads, k, n] -> [batch, heads, m, n]
// This is the exact pattern for Q @ K^T and attn @ V in attention

let batch = 1;
let heads = 12;  // Number of attention heads
let seq_len = 512;
let head_dim = 64;

// Q: [batch, heads, seq_len, head_dim]
let q_data: Vec<f32> = vec![0.0; batch * heads * seq_len * head_dim];
// K^T: [batch, heads, head_dim, seq_len] (already transposed)
let kt_data: Vec<f32> = vec![0.0; batch * heads * head_dim * seq_len];

// Compute attention scores: Q @ K^T
let attn_scores = Matrix::batched_matmul_4d(
    &q_data,
    &kt_data,
    batch,
    heads,
    seq_len,   // m
    head_dim,  // k
    seq_len,   // n
)?;
// Result: [batch, heads, seq_len, seq_len] attention scores
```

This is critical for transformer performance - each (batch, head) pair is processed independently using SIMD matmul.

## GPU Acceleration

For large matrices, use the GPU backend.

```rust
use trueno::GpuBackend;

let mut gpu = GpuBackend::new();
let a = Matrix::<f32>::from_vec(1024, 1024, /* ... */)?;
let b = Matrix::<f32>::from_vec(1024, 1024, /* ... */)?;

// GPU-accelerated matmul
let c = gpu.matmul(&a, &b)?;
```

## Performance Tips

1. **Matrix multiplication**: O(n³) - GPU beneficial for n > 500
2. **Convolution**: Use separable kernels when possible
3. **Memory layout**: Row-major storage for cache efficiency
4. **Batch operations**: Group small matrices for GPU efficiency

See the [GPU Performance Guide](../performance/gpu-performance.md) for details.

# Vector Math

This chapter demonstrates Trueno's vector math capabilities using the `quickstart` and `performance_demo` examples.

## Quick Start

Run the quickstart example to see all core vector operations:

```bash
cargo run --example quickstart
```

### Basic Operations

```rust
use trueno::Vector;

// Create vectors
let a = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
let b = Vector::from_slice(&[5.0, 6.0, 7.0, 8.0]);

// Element-wise operations
let sum = a.add(&b)?;      // [6.0, 8.0, 10.0, 12.0]
let prod = a.mul(&b)?;     // [5.0, 12.0, 21.0, 32.0]

// Reductions
let dot = a.dot(&b)?;      // 70.0
let norm = a.norm_l2()?;   // 5.477...

// Statistical operations
let mean = a.mean()?;      // 2.5
let variance = a.variance()?;
```

### Backend Selection

Trueno automatically selects the best available backend:

```rust
use trueno::{Vector, Backend};

// Auto backend (recommended)
let v = Vector::from_slice(&data);

// Force specific backend
let scalar = Vector::from_slice_with_backend(&data, Backend::Scalar);
```

## Performance Comparison

Run the performance demo to see SIMD speedups:

```bash
cargo run --release --example performance_demo
```

### Expected Results

| Operation | SIMD Speedup | Notes |
|-----------|-------------|-------|
| Dot Product | 3-4x | Compute-intensive |
| Sum Reduction | 3x | Compute-intensive |
| Max Finding | 3x | Compute-intensive |
| Element-wise Add | 1.5x | Memory-bound |
| Element-wise Mul | 1.5x | Memory-bound |

### Understanding the Results

**Compute-intensive operations** (dot product, sum, max) show significant speedups because SIMD can process 8 f32 values simultaneously.

**Memory-bound operations** (add, mul) show modest speedups because performance is limited by memory bandwidth, not computation.

## ML Similarity Operations

Run the similarity example:

```bash
cargo run --example ml_similarity
```

### Cosine Similarity

```rust
use trueno::Vector;

let query = Vector::from_slice(&[0.5, 0.8, 0.2]);
let document = Vector::from_slice(&[0.6, 0.7, 0.3]);

// Compute cosine similarity
let norm_q = query.norm_l2()?;
let norm_d = document.norm_l2()?;
let dot = query.dot(&document)?;
let similarity = dot / (norm_q * norm_d);
```

### k-NN Classification

```rust
// Compute Euclidean distances
let diff = query.sub(&sample)?;
let dist_sq = diff.dot(&diff)?;
let distance = dist_sq.sqrt();
```

## Layer Normalization

```rust
use trueno::Vector;

let input = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);

// Compute mean and variance
let mean = input.mean()?;
let centered = input.sub_scalar(mean)?;
let var = centered.dot(&centered)? / input.len() as f32;
let std = (var + 1e-5).sqrt();

// Normalize
let normalized = centered.mul_scalar(1.0 / std)?;
```

## See Also

- [Performance Demo](../performance/benchmarks.md) - Detailed benchmarks
- [ML Similarity](./neural-networks.md) - ML-specific operations
- [Backend Selection](../architecture/backend-selection.md) - How backends are chosen

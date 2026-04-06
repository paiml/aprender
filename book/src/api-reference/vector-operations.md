# Vector Operations

The `Vector<T>` type is the core data structure in Trueno, providing SIMD-accelerated operations on contiguous arrays of floating-point numbers.

## Creating Vectors

```rust
use trueno::{Vector, Backend};

// From a slice (uses best available backend)
let v = Vector::<f32>::from_slice(&[1.0, 2.0, 3.0, 4.0]);

// With explicit backend
let v_scalar = Vector::<f32>::from_slice_with_backend(
    &[1.0, 2.0, 3.0],
    Backend::Scalar
);

// From Vec
let v = Vector::<f32>::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
```

## Element-wise Operations

All element-wise operations return a new `Vector` with the same length.

```rust
let a = Vector::<f32>::from_slice(&[1.0, 2.0, 3.0]);
let b = Vector::<f32>::from_slice(&[4.0, 5.0, 6.0]);

// Arithmetic
let sum = a.add(&b)?;      // [5.0, 7.0, 9.0]
let diff = a.sub(&b)?;     // [-3.0, -3.0, -3.0]
let prod = a.mul(&b)?;     // [4.0, 10.0, 18.0]
let quot = a.div(&b)?;     // [0.25, 0.4, 0.5]

// Scalar operations
let scaled = a.scale(2.0)?; // [2.0, 4.0, 6.0]

// Math functions
let sqrts = a.sqrt()?;
let exps = a.exp()?;
let logs = a.ln()?;
```

## Reduction Operations

Reductions collapse a vector to a single value.

```rust
let v = Vector::<f32>::from_slice(&[1.0, 2.0, 3.0, 4.0]);

let total = v.sum()?;        // 10.0
let maximum = v.max()?;      // 4.0
let minimum = v.min()?;      // 1.0
let dot = a.dot(&b)?;        // Dot product

// Norms
let l1 = v.norm_l1()?;       // Manhattan norm
let l2 = v.norm_l2()?;       // Euclidean norm
let linf = v.norm_linf()?;   // Max absolute value

// Argmax/Argmin
let idx_max = v.argmax()?;   // Index of max element
let idx_min = v.argmin()?;   // Index of min element
```

## Activation Functions

Common neural network activations, optimized for ML inference.

```rust
let x = Vector::<f32>::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0]);

// Classic activations
let relu = x.relu()?;
let sigmoid = x.sigmoid()?;
let tanh_v = x.tanh_activation()?;

// Modern activations (Transformer era)
let gelu = x.gelu()?;       // BERT, GPT
let swish = x.swish()?;     // EfficientNet
let mish = x.mish()?;       // YOLOv4

// Variants
let leaky = x.leaky_relu(0.01)?;
let elu = x.elu(1.0)?;
let selu = x.selu()?;
```

## Layer Normalization

For transformer architectures.

```rust
let hidden = Vector::<f32>::from_slice(&[1.0, 2.0, 3.0, 4.0]);
let gamma = Vector::<f32>::from_slice(&[1.0, 1.0, 1.0, 1.0]); // scale
let beta = Vector::<f32>::from_slice(&[0.0, 0.0, 0.0, 0.0]);  // shift

let normalized = hidden.layer_norm(&gamma, &beta, 1e-5)?;
// Output has mean ≈ 0, variance ≈ 1
```

## Similarity Metrics

For ML applications like recommendation systems.

```rust
let a = Vector::<f32>::from_slice(&[1.0, 2.0, 3.0]);
let b = Vector::<f32>::from_slice(&[4.0, 5.0, 6.0]);

let cosine = a.cosine_similarity(&b)?;  // [-1, 1]
let euclidean = a.euclidean_distance(&b)?;
let manhattan = a.manhattan_distance(&b)?;
```

## Backend Selection

Vectors automatically use the best available SIMD backend.

```rust
use trueno::{select_best_available_backend, OperationType};

// Check what's available
let backend = select_best_available_backend();
println!("Using: {:?}", backend); // e.g., AVX2

// Operation-aware selection (memory-bound vs compute-bound)
let mem_backend = select_backend_for_operation(OperationType::MemoryBound);
let compute_backend = select_backend_for_operation(OperationType::ComputeBound);
```

## Performance Characteristics

| Operation | Type | Expected Speedup |
|-----------|------|------------------|
| `dot` | Compute-bound | 11-12x (AVX-512) |
| `sum`, `max`, `min` | Compute-bound | 4-8x |
| `add`, `mul` | Memory-bound | 1-2x |
| `relu`, `sigmoid` | Mixed | 2-4x |

See [Performance Guide](../performance/optimization-guide.md) for detailed analysis.

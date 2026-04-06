# NumPy to Trueno Conversion

Batuta's `NumPyConverter` maps NumPy operations to their `trueno` equivalents.
Trueno provides SIMD-accelerated (AVX2, AVX-512, NEON) implementations that
match NumPy semantics while eliminating the Python interpreter overhead.

## Array Creation

**Python (NumPy)**

```python
import numpy as np

a = np.array([1.0, 2.0, 3.0])
b = np.zeros(1024)
c = np.ones((4, 4))
```

**Rust (Trueno)**

```rust
use trueno::Vector;

let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
let b = Vector::zeros(1024);
let c = Matrix::ones(4, 4);
```

Trueno's `Vector::from_slice` is the direct equivalent of `np.array` for 1-D
data. For 2-D data, `Matrix::from_slice` accepts row-major layout, matching
NumPy's default C-order.

## Element-wise Operations

**Python (NumPy)**

```python
c = np.add(a, b)       # or a + b
d = np.multiply(a, b)  # or a * b
e = np.subtract(a, b)  # or a - b
```

**Rust (Trueno)**

```rust
let c = a.add(&b).unwrap();
let d = a.mul(&b).unwrap();
let e = a.sub(&b).unwrap();
```

Operations return `Result` because trueno validates shape compatibility at
runtime. Dimension mismatches produce a clear error instead of silent
broadcasting bugs.

## Dot Product and Matrix Multiply

**Python (NumPy)**

```python
dot = np.dot(a, b)         # Vector dot product
result = np.matmul(X, W)   # Matrix multiply, or X @ W
```

**Rust (Trueno)**

```rust
let dot = a.dot(&b).unwrap();
let result = x.matmul(&w).unwrap();
```

Dot products and matrix multiplies are classified as high-complexity operations.
Batuta's MoE backend selector routes them to GPU when data exceeds the PCIe
5x transfer cost threshold (typically above 50,000 elements).

## Reductions

**Python (NumPy)**

```python
total = np.sum(a)
avg = np.mean(a)
maximum = np.max(a)
```

**Rust (Trueno)**

```rust
let total = a.sum();
let avg = a.mean();
let maximum = a.max();
```

Reductions are medium-complexity operations. For vectors above roughly 10,000
elements, trueno automatically dispatches to SIMD kernels (AVX2 on x86_64,
NEON on aarch64).

## Broadcasting Semantics

NumPy broadcasting rules are preserved in trueno. A scalar broadcast across a
vector works identically:

```python
# NumPy: scalar broadcast
scaled = a * 2.0
```

```rust
// Trueno: scalar broadcast
let scaled = a.scale(2.0);
```

For shape-incompatible operations, trueno returns an error rather than silently
expanding dimensions. This catches a common class of NumPy bugs at the point of
failure instead of producing wrong results downstream.

## Backend Selection

Batuta assigns each NumPy operation a complexity tier and selects the optimal
backend based on data size:

| Operation    | Complexity | Small Data | Large Data |
|-------------|------------|------------|------------|
| add, mul    | Low        | Scalar     | SIMD       |
| sum, mean   | Medium     | Scalar     | SIMD       |
| dot, matmul | High       | SIMD       | GPU        |

This selection happens automatically during the Optimize phase. No manual
annotation is required.

## Key Takeaways

- `np.array` maps to `Vector::from_slice` or `Matrix::from_slice`.
- Element-wise operations return `Result` for shape safety.
- Dot products and matrix multiplies get automatic GPU acceleration for large
  data via the MoE backend selector.
- Broadcasting semantics are preserved; shape mismatches become explicit errors.
- SIMD acceleration is transparent -- trueno selects the best instruction set
  available on the target CPU at runtime.

---

**Navigate:** [Table of Contents](../SUMMARY.md)

# Semantic Preservation

**Semantic Preservation** is Batuta's core guarantee: transpiled Rust code produces results identical to the original source.

## Core Principle

> Correctness is non-negotiable. A transpilation that changes behavior is worse than no transpilation at all.

Every pipeline execution validates that the output program is semantically equivalent to the input, across numerical results, API behavior, and system interactions.

## Three Pillars

### 1. Numerical Fidelity

Floating-point operations must produce bitwise-identical or epsilon-bounded results:

```rust
// Python: numpy.dot(a, b)
// Rust:   trueno::simd::dot(a, b)

// Validation: compare outputs within machine epsilon
fn verify_numerical_fidelity(python_out: &[f64], rust_out: &[f64]) -> bool {
    python_out.iter().zip(rust_out).all(|(p, r)| {
        (p - r).abs() < f64::EPSILON * 10.0
    })
}
```

### 2. API Equivalence

Public interfaces must accept the same inputs and produce the same outputs:

| Python | Rust (Transpiled) | Guarantee |
|--------|-------------------|-----------|
| `sklearn.fit(X, y)` | `aprender::fit(&x, &y)` | Same model weights |
| `numpy.linalg.svd(A)` | `trueno::linalg::svd(&a)` | Same decomposition |
| `torch.inference(x)` | `realizar::infer(&x)` | Same predictions |

### 3. Behavioral Parity

Side effects — file I/O, network calls, exit codes — must match:

```bash
# Validate behavioral parity via syscall tracing
batuta validate --trace

# Renacer captures syscalls from both programs
# Python run:  open("out.csv", W) → write(1024 bytes) → close()
# Rust run:    open("out.csv", W) → write(1024 bytes) → close()
# Result: MATCH
```

## Validation Pipeline

Batuta's Phase 4 (Validation) enforces semantic preservation automatically:

```
Source Program ──► Run + Capture ──► Reference Output
                                          │
                                    ┌─────┴─────┐
                                    │  Compare   │
                                    └─────┬─────┘
                                          │
Transpiled Rust ──► Run + Capture ──► Actual Output
```

## Example: NumPy to Trueno

```python
# Original Python
import numpy as np
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
result = np.dot(a, b)  # 32.0
```

```rust
// Transpiled Rust — semantically identical
use trueno::Tensor;
let a = Tensor::from_slice(&[1.0, 2.0, 3.0]);
let b = Tensor::from_slice(&[4.0, 5.0, 6.0]);
let result = a.dot(&b);  // 32.0
```

Batuta validates that both produce `32.0` before marking the transpilation as successful.

## Benefits

1. **Confidence** - Teams trust that transpiled code is correct
2. **Automation** - No manual verification needed
3. **Regression prevention** - Every change is validated against the reference
4. **Auditability** - Syscall traces provide a provable equivalence record

---

**Navigate:** [Table of Contents](../SUMMARY.md)

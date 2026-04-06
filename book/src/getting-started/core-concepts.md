# Core Concepts

Understanding Trueno's fundamental concepts will help you write efficient, safe code.

## The Vector Type

`Vector<T>` is Trueno's core abstraction:

```rust
use trueno::Vector;

let v = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
```

**Key properties:**
- Generic over numeric types: `f32`, `f64`, `i32`, `i64`
- Immutable by default (functional style)
- Backend selected at creation time (no repeated detection)
- Zero-copy views with `as_slice()`

## Backend Selection

Trueno automatically selects the best backend when you create a `Vector`:

```rust
// Automatic backend selection
let v = Vector::from_slice(&[1.0; 1000]);
println!("{:?}", v.backend());  // Avx2, Sse2, Neon, etc.

// Manual backend override (for testing/profiling)
let v = Vector::with_backend(&[1.0; 1000], Backend::Scalar);
```

**Selection priority:**
1. GPU (if workload >100K elements and GPU available)
2. AVX-512 (if CPU supports)
3. AVX2 (if CPU supports)
4. AVX (if CPU supports)
5. SSE2 (x86_64 baseline)
6. NEON (ARM64)
7. Scalar fallback

## Safety Model

Trueno maintains safety through three layers:

### Layer 1: Type System

```rust
// Compile-time type safety
let a = Vector::from_slice(&[1.0f32, 2.0, 3.0]);
let b = Vector::from_slice(&[4.0f64, 5.0, 6.0]);

// ❌ Compile error: type mismatch
// let result = a.add(&b);
```

### Layer 2: Runtime Validation

```rust
// Runtime size checking
let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
let b = Vector::from_slice(&[4.0, 5.0]);

// Returns Err(SizeMismatch)
let result = a.add(&b);
```

### Layer 3: Unsafe Isolation

All `unsafe` code is isolated to backend implementations:

```rust
// ✅ 100% safe public API
pub fn add(&self, other: &Self) -> Result<Self> {
    validate_sizes(self, other)?;  // Safe
    
    match self.backend {
        Backend::Avx2 => unsafe { self.add_avx2(other) },  // ❌ Unsafe (internal only)
        Backend::Scalar => self.add_scalar(other),  // ✅ Safe
    }
}
```

## Error Handling

Trueno uses Rust's `Result` type for robust error handling:

```rust
use trueno::{Vector, TruenoError};

fn process_vectors() -> Result<Vector, TruenoError> {
    let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
    let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
    
    let sum = a.add(&b)?;  // Propagate errors with ?
    let product = sum.mul_scalar(2.0)?;
    
    Ok(product)
}
```

**Error types:**
- `SizeMismatch` - Vectors have incompatible sizes
- `BackendError` - Backend initialization failed
- `GpuError` - GPU operation failed
- `InvalidInput` - Invalid parameters (NaN, infinity)

## Performance Model

Understanding Trueno's performance characteristics helps you write efficient code.

### Operation Complexity

Operations fall into three categories:

**Low complexity** (add, sub, mul, div):
- Prefer SIMD for >1K elements
- Memory-bandwidth limited
- Expect 1.1-2x speedup

**Medium complexity** (dot, sum, max):
- SIMD shines here (3-5x speedup)
- Compute-bound, not memory-bound
- Use SIMD even for 100 elements

**High complexity** (tanh, exp, log):
- Excellent SIMD performance (6-9x speedup)
- Compute-intensive operations
- Consider GPU for >100K elements

### Backend Overhead

Each backend has different overhead characteristics:

| Backend | Overhead | Best For |
|---------|----------|----------|
| Scalar | None | <100 elements, testing |
| SSE2 | ~20ns | 100-100K elements |
| AVX2 | ~30ns | 1K-100K elements |
| GPU | ~0.5ms | >100K elements |

## Next Steps

- **[First Program](./first-program.md)** - Build a complete example
- **[Architecture Overview](../architecture/overview.md)** - Deep dive into backends
- **[API Reference](../api-reference/vector-operations.md)** - Explore all operations

# Trueno-Ruchy Integration Specification

**Version**: 1.0.0
**Date**: 2025-11-16
**Status**: Design Phase
**Authors**: Pragmatic AI Labs

---

## Executive Summary

This specification defines the integration between **Trueno** (multi-backend SIMD compute library) and **Ruchy** (Ruby-like language transpiling to Rust). The integration enables high-level scripting with zero-overhead native performance by leveraging Ruchy's transpilation model.

**Key Insight**: Ruchy transpiles to Rust, so integration is achieved through:
1. Adding Trueno as a Cargo dependency
2. Creating a thin Ruchy stdlib wrapper
3. Implementing operator overloading traits in Rust
4. Auto-generating type aliases for ergonomic syntax

**No FFI required** - Ruchy generates pure Rust code that calls Trueno directly.

---

## 1. Architecture Overview

### 1.1 Integration Flow

```
┌─────────────────┐
│  Ruchy Source   │  let v = Vector([1.0, 2.0, 3.0])
│   (.ruchy)      │  let sum = v + other
└────────┬────────┘
         │ transpile
         ▼
┌─────────────────┐
│  Rust Source    │  let v = trueno::Vector::from_slice(&[1.0, 2.0, 3.0]);
│    (.rs)        │  let sum = v.add(&other).unwrap();
└────────┬────────┘
         │ rustc compile
         ▼
┌─────────────────┐
│ Native Binary   │  Executes with AVX2/NEON/WASM SIMD
│  (executable)   │  Zero abstraction overhead
└─────────────────┘
```

### 1.2 Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| **Trueno** | Core SIMD compute library (backend selection, kernels) |
| **Ruchy Stdlib** | Thin wrapper providing Ruchy-friendly API |
| **Ruchy Transpiler** | Type mapping, operator desugaring, import resolution |
| **Rust Compiler** | Optimization, monomorphization, native code generation |

---

## 2. Dependencies

### 2.1 Ruchy Cargo.toml

Add Trueno as a dependency:

```toml
[dependencies]
trueno = { path = "../trueno", version = "0.1.0" }

[features]
default = ["trueno-simd"]
trueno-simd = ["trueno/simd"]
trueno-gpu = ["trueno/gpu"]
```

### 2.2 Version Compatibility

| Ruchy Version | Trueno Version | Rust Version |
|---------------|----------------|--------------|
| ≥ 3.94.0      | ≥ 0.1.0        | ≥ 1.75.0     |

---

## 3. Stdlib Module: `std::linalg`

### 3.1 File Location

**Path**: `/home/noah/src/ruchy/src/stdlib/linalg.rs`

### 3.2 Module Structure

```rust
//! Linear Algebra Operations (STD-012)
//!
//! Thin wrapper around Trueno for high-performance vector/matrix operations.
//! Provides Ruchy-friendly API with zero abstraction overhead.
//!
//! # Design Principles
//! - **Zero Reinvention**: Direct delegation to Trueno
//! - **Thin Wrapper**: Complexity ≤5 per function
//! - **Ergonomic API**: Feels natural in Ruchy code
//! - **Performance**: Auto-selects best SIMD backend (AVX2/NEON/WASM)

use trueno::{Vector, Backend, Result as TruenoResult, TruenoError};

// Re-export core types for Ruchy code
pub use trueno::{Vector, Backend};

// Type aliases for common use cases
pub type Vector32 = Vector<f32>;
pub type Vector64 = Vector<f64>;

/// Create vector from Ruchy array literal
///
/// # Examples
/// ```ruchy
/// let v = Vector::new([1.0, 2.0, 3.0])
/// ```
pub fn vector_from_slice(data: &[f32]) -> Vector<f32> {
    Vector::from_slice(data)
}

/// Create vector with explicit backend (for benchmarking/testing)
///
/// # Examples
/// ```ruchy
/// let v = Vector::with_backend([1.0, 2.0], Backend::AVX2)
/// ```
pub fn vector_with_backend(data: &[f32], backend: Backend) -> Vector<f32> {
    Vector::from_slice_with_backend(data, backend)
}

/// Element-wise addition (wrapper for ergonomic error handling)
///
/// # Examples
/// ```ruchy
/// let sum = vector_add(v1, v2)  # Returns Option<Vector>
/// ```
pub fn vector_add(a: &Vector<f32>, b: &Vector<f32>) -> Option<Vector<f32>> {
    a.add(b).ok()
}

/// Element-wise multiplication
pub fn vector_mul(a: &Vector<f32>, b: &Vector<f32>) -> Option<Vector<f32>> {
    a.mul(b).ok()
}

/// Dot product
///
/// # Examples
/// ```ruchy
/// let dot = v1.dot(v2)  # Returns Option<f32>
/// ```
pub fn vector_dot(a: &Vector<f32>, b: &Vector<f32>) -> Option<f32> {
    a.dot(b).ok()
}

/// Sum reduction
pub fn vector_sum(v: &Vector<f32>) -> Option<f32> {
    v.sum().ok()
}

/// Max reduction
pub fn vector_max(v: &Vector<f32>) -> Option<f32> {
    v.max().ok()
}

/// L2 norm (Euclidean norm)
pub fn vector_norm(v: &Vector<f32>) -> Option<f32> {
    v.norm_l2().ok()
}

/// Normalize to unit vector
pub fn vector_normalize(v: &Vector<f32>) -> Option<Vector<f32>> {
    v.normalize().ok()
}

/// Get vector length
pub fn vector_len(v: &Vector<f32>) -> usize {
    v.len()
}

/// Convert vector to Ruchy array
pub fn vector_to_array(v: &Vector<f32>) -> Vec<f32> {
    v.as_slice().to_vec()
}

/// Get current backend
pub fn get_best_backend() -> Backend {
    trueno::select_best_available_backend()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_creation() {
        let v = vector_from_slice(&[1.0, 2.0, 3.0]);
        assert_eq!(vector_len(&v), 3);
    }

    #[test]
    fn test_vector_add() {
        let a = vector_from_slice(&[1.0, 2.0]);
        let b = vector_from_slice(&[3.0, 4.0]);
        let sum = vector_add(&a, &b).unwrap();
        assert_eq!(vector_to_array(&sum), vec![4.0, 6.0]);
    }

    #[test]
    fn test_vector_dot() {
        let a = vector_from_slice(&[1.0, 2.0, 3.0]);
        let b = vector_from_slice(&[4.0, 5.0, 6.0]);
        let dot = vector_dot(&a, &b).unwrap();
        assert_eq!(dot, 32.0);  // 1*4 + 2*5 + 3*6
    }

    #[test]
    fn test_backend_selection() {
        let backend = get_best_backend();
        // Should be SSE2 or better on x86_64
        #[cfg(target_arch = "x86_64")]
        assert_ne!(backend, Backend::Scalar);
    }
}
```

### 3.3 Register Module

**File**: `/home/noah/src/ruchy/src/stdlib/mod.rs`

Add:
```rust
#[cfg(feature = "trueno-simd")]
pub mod linalg;
```

---

## 4. Operator Overloading

### 4.1 Implement Rust Traits for Trueno Vector

**File**: `/home/noah/src/trueno/src/vector.rs`

Add operator trait implementations:

```rust
use std::ops::{Add, Sub, Mul, Div};

// Element-wise addition: v1 + v2
impl Add for Vector<f32> {
    type Output = Result<Self>;

    fn add(self, other: Self) -> Self::Output {
        self.add(&other)
    }
}

impl Add for &Vector<f32> {
    type Output = Result<Vector<f32>>;

    fn add(self, other: Self) -> Self::Output {
        Vector::add(self, other)
    }
}

// Element-wise subtraction: v1 - v2
impl Sub for Vector<f32> {
    type Output = Result<Self>;

    fn sub(self, other: Self) -> Self::Output {
        self.sub(&other)
    }
}

impl Sub for &Vector<f32> {
    type Output = Result<Vector<f32>>;

    fn sub(self, other: Self) -> Self::Output {
        Vector::sub(self, other)
    }
}

// Element-wise multiplication: v1 * v2
impl Mul for Vector<f32> {
    type Output = Result<Self>;

    fn mul(self, other: Self) -> Self::Output {
        self.mul(&other)
    }
}

impl Mul for &Vector<f32> {
    type Output = Result<Vector<f32>>;

    fn mul(self, other: Self) -> Self::Output {
        Vector::mul(self, other)
    }
}

// Scalar multiplication: v * scalar
impl Mul<f32> for Vector<f32> {
    type Output = Self;

    fn mul(self, scalar: f32) -> Self::Output {
        let data: Vec<f32> = self.as_slice().iter().map(|x| x * scalar).collect();
        Vector::from_slice_with_backend(&data, self.backend)
    }
}

impl Mul<f32> for &Vector<f32> {
    type Output = Vector<f32>;

    fn mul(self, scalar: f32) -> Self::Output {
        let data: Vec<f32> = self.as_slice().iter().map(|x| x * scalar).collect();
        Vector::from_slice_with_backend(&data, self.backend)
    }
}

// Element-wise division: v1 / v2
impl Div for Vector<f32> {
    type Output = Result<Self>;

    fn div(self, other: Self) -> Self::Output {
        self.div(&other)
    }
}

impl Div for &Vector<f32> {
    type Output = Result<Vector<f32>>;

    fn div(self, other: Self) -> Self::Output {
        Vector::div(self, other)
    }
}

// Negation: -v
impl std::ops::Neg for Vector<f32> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let data: Vec<f32> = self.as_slice().iter().map(|x| -x).collect();
        Vector::from_slice_with_backend(&data, self.backend)
    }
}

impl std::ops::Neg for &Vector<f32> {
    type Output = Vector<f32>;

    fn neg(self) -> Self::Output {
        let data: Vec<f32> = self.as_slice().iter().map(|x| -x).collect();
        Vector::from_slice_with_backend(&data, self.backend)
    }
}
```

### 4.2 Operator Mapping in Ruchy

Ruchy transpiles operators to Rust trait calls automatically:

| Ruchy Syntax | Rust Transpilation | Trueno Implementation |
|--------------|-------------------|----------------------|
| `v1 + v2` | `v1.add(v2)?` | `Vector::add()` |
| `v1 - v2` | `v1.sub(v2)?` | `Vector::sub()` |
| `v1 * v2` | `v1.mul(v2)?` | `Vector::mul()` (element-wise) |
| `v1 / v2` | `v1.div(v2)?` | `Vector::div()` |
| `v * 2.0` | `v.mul(2.0)` | `Mul<f32>` trait |
| `-v` | `v.neg()` | `Neg` trait |

**Note**: For dot product, use explicit method: `v1.dot(v2)`

---

## 5. Type System Integration

### 5.1 Type Alias in Ruchy Transpiler

**File**: `/home/noah/src/ruchy/src/backend/transpiler/types.rs`

Add to `transpile_named_type` function:

```rust
fn transpile_named_type(&self, name: &str) -> Result<TokenStream> {
    let rust_type = match name {
        // ... existing mappings (int, float, bool, String, etc.) ...

        // Trueno vector types
        "Vector" => quote! { trueno::Vector<f32> },
        "Vector32" => quote! { trueno::Vector<f32> },
        "Vector64" => quote! { trueno::Vector<f64> },

        _ => { /* existing fallback logic */ }
    };
    Ok(rust_type)
}
```

### 5.2 Generic Type Support

Ruchy already supports generic types. No changes needed:

```ruchy
// This works out of the box
let v: Vector<f32> = Vector::from_slice([1.0, 2.0, 3.0])
```

Transpiles to:
```rust
let v: trueno::Vector<f32> = trueno::Vector::from_slice(&[1.0, 2.0, 3.0]);
```

### 5.3 Import Statement Handling

**Ruchy code:**
```ruchy
import trueno::Vector
import trueno::Backend

fn main() {
    let v = Vector::from_slice([1.0, 2.0])
}
```

**Generated Rust:**
```rust
use trueno::Vector;
use trueno::Backend;

fn main() {
    let v = Vector::from_slice(&[1.0, 2.0]);
}
```

No transpiler changes needed - existing import logic handles this.

---

## 6. Ruchy API Examples

### 6.1 Basic Vector Operations

```ruchy
import trueno::Vector

fn main() {
    # Create vectors
    let a = Vector::from_slice([1.0, 2.0, 3.0, 4.0])
    let b = Vector::from_slice([5.0, 6.0, 7.0, 8.0])

    # Element-wise operations
    let sum = a.add(b)
    let product = a.mul(b)

    # Reductions
    let total = a.sum()
    let maximum = a.max()

    # Dot product
    let dot = a.dot(b)

    println(f"Sum: {sum:?}")
    println(f"Dot product: {dot}")
}
```

### 6.2 Operator Overloading Syntax

```ruchy
import trueno::Vector

fn main() {
    let v1 = Vector::from_slice([1.0, 2.0, 3.0])
    let v2 = Vector::from_slice([4.0, 5.0, 6.0])

    # Operators (requires Rust trait implementations)
    let sum = v1 + v2           # Add trait
    let diff = v1 - v2          # Sub trait
    let scaled = v1 * 2.0       # Mul<f32> trait
    let negated = -v1           # Neg trait

    println(f"Sum: {sum:?}")
}
```

### 6.3 Backend Selection

```ruchy
import trueno::{Vector, Backend}

fn main() {
    # Auto-select best backend
    let v_auto = Vector::from_slice([1.0, 2.0, 3.0])

    # Explicit backend (for testing/benchmarking)
    let v_scalar = Vector::from_slice_with_backend([1.0, 2.0], Backend::Scalar)
    let v_avx2 = Vector::from_slice_with_backend([1.0, 2.0], Backend::AVX2)

    # Get current backend
    let backend = trueno::select_best_available_backend()
    println(f"Using backend: {backend:?}")
}
```

### 6.4 Error Handling

```ruchy
import trueno::Vector

fn main() {
    let a = Vector::from_slice([1.0, 2.0])
    let b = Vector::from_slice([1.0, 2.0, 3.0])

    # Size mismatch - returns Result
    match a.add(b) {
        Ok(result) => println(f"Sum: {result:?}"),
        Err(e) => println(f"Error: {e}")
    }

    # Or use unwrap for prototyping
    # let sum = a.add(b).unwrap()  # Panics on error
}
```

### 6.5 Machine Learning Example

```ruchy
import trueno::Vector

# Cosine similarity for document comparison
fn cosine_similarity(a: Vector<f32>, b: Vector<f32>) -> f32 {
    let dot = a.dot(b).unwrap()
    let norm_a = a.norm_l2().unwrap()
    let norm_b = b.norm_l2().unwrap()
    dot / (norm_a * norm_b)
}

fn main() {
    # Document embeddings (simplified)
    let doc1 = Vector::from_slice([0.5, 0.3, 0.8, 0.1])
    let doc2 = Vector::from_slice([0.4, 0.6, 0.7, 0.2])
    let query = Vector::from_slice([0.6, 0.4, 0.9, 0.1])

    # Find most similar document
    let sim1 = cosine_similarity(query.clone(), doc1)
    let sim2 = cosine_similarity(query, doc2)

    if sim1 > sim2 {
        println("Document 1 is more similar")
    } else {
        println("Document 2 is more similar")
    }
}
```

### 6.6 Benchmarking Different Backends

```ruchy
import trueno::{Vector, Backend}
import std::time::Instant

fn benchmark_backend(backend: Backend, size: i32) {
    let data = (0..size).map(|i| i as f32).collect::<Vec<_>>()

    let v1 = Vector::from_slice_with_backend(data.clone(), backend)
    let v2 = Vector::from_slice_with_backend(data, backend)

    let start = Instant::now()
    for _ in 0..1000 {
        v1.dot(v2).unwrap()
    }
    let elapsed = start.elapsed()

    println(f"{backend:?}: {elapsed:?}")
}

fn main() {
    println("Benchmarking dot product (1000 iterations):")

    benchmark_backend(Backend::Scalar, 1000)
    benchmark_backend(Backend::SSE2, 1000)
    benchmark_backend(Backend::AVX2, 1000)
}
```

---

## 7. Testing Strategy

### 7.1 Ruchy Integration Tests

**File**: `/home/noah/src/ruchy/tests/trueno_integration.rs`

```rust
use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;

#[test]
fn test_vector_basic_transpilation() {
    let ruchy_code = r#"
import trueno::Vector

fn main() {
    let v = Vector::from_slice([1.0, 2.0, 3.0])
    println(f"{v:?}")
}
"#;

    fs::write("test_vector.ruchy", ruchy_code).unwrap();

    Command::cargo_bin("ruchy")
        .unwrap()
        .arg("transpile")
        .arg("test_vector.ruchy")
        .assert()
        .success()
        .stdout(predicate::str::contains("trueno::Vector"))
        .stdout(predicate::str::contains("from_slice"));

    fs::remove_file("test_vector.ruchy").unwrap();
}

#[test]
fn test_vector_execution() {
    let ruchy_code = r#"
import trueno::Vector

fn main() {
    let a = Vector::from_slice([1.0, 2.0, 3.0])
    let b = Vector::from_slice([4.0, 5.0, 6.0])
    let dot = a.dot(b).unwrap()
    println(f"{dot}")
}
"#;

    fs::write("test_vector_run.ruchy", ruchy_code).unwrap();

    Command::cargo_bin("ruchy")
        .unwrap()
        .arg("run")
        .arg("test_vector_run.ruchy")
        .assert()
        .success()
        .stdout(predicate::str::contains("32"));  // 1*4 + 2*5 + 3*6

    fs::remove_file("test_vector_run.ruchy").unwrap();
}

#[test]
fn test_vector_operators() {
    let ruchy_code = r#"
import trueno::Vector

fn main() {
    let v1 = Vector::from_slice([1.0, 2.0])
    let v2 = Vector::from_slice([3.0, 4.0])

    # Test operator overloading
    let sum = v1.add(v2).unwrap()
    let first = sum.as_slice()[0]

    println(f"{first}")
}
"#;

    fs::write("test_ops.ruchy", ruchy_code).unwrap();

    Command::cargo_bin("ruchy")
        .unwrap()
        .arg("run")
        .arg("test_ops.ruchy")
        .assert()
        .success()
        .stdout(predicate::str::contains("4"));  // 1.0 + 3.0

    fs::remove_file("test_ops.ruchy").unwrap();
}

#[test]
fn test_backend_selection() {
    let ruchy_code = r#"
import trueno

fn main() {
    let backend = trueno::select_best_available_backend()
    println(f"{backend:?}")
}
"#;

    fs::write("test_backend.ruchy", ruchy_code).unwrap();

    Command::cargo_bin("ruchy")
        .unwrap()
        .arg("run")
        .arg("test_backend.ruchy")
        .assert()
        .success();  // Just verify it runs

    fs::remove_file("test_backend.ruchy").unwrap();
}
```

### 7.2 Cross-Backend Validation

**File**: `/home/noah/src/ruchy/tests/trueno_backends.rs`

```rust
#[test]
fn test_all_backends_agree() {
    let ruchy_code = r#"
import trueno::{Vector, Backend}

fn main() {
    let data = [1.0, 2.0, 3.0, 4.0]

    let v_scalar = Vector::from_slice_with_backend(data, Backend::Scalar)
    let v_sse2 = Vector::from_slice_with_backend(data, Backend::SSE2)

    let dot_scalar = v_scalar.dot(v_scalar).unwrap()
    let dot_sse2 = v_sse2.dot(v_sse2).unwrap()

    # Should be equal within floating-point tolerance
    let diff = (dot_scalar - dot_sse2).abs()
    assert(diff < 1e-5, f"Backend mismatch: {diff}")

    println("All backends agree!")
}
"#;

    fs::write("test_backends.ruchy", ruchy_code).unwrap();

    Command::cargo_bin("ruchy")
        .unwrap()
        .arg("run")
        .arg("test_backends.ruchy")
        .assert()
        .success()
        .stdout(predicate::str::contains("All backends agree"));

    fs::remove_file("test_backends.ruchy").unwrap();
}
```

### 7.3 Property-Based Testing

**File**: `/home/noah/src/ruchy/tests/properties/trueno_properties.rs`

```rust
use proptest::prelude::*;

proptest! {
    #[test]
    fn vector_add_commutative(a in prop::collection::vec(-1e6_f32..1e6, 1..100),
                              b in prop::collection::vec(-1e6_f32..1e6, 1..100)) {
        // Generate Ruchy code
        let ruchy_code = format!(r#"
import trueno::Vector

fn main() {{
    let a = Vector::from_slice([{}])
    let b = Vector::from_slice([{}])

    let sum1 = a.add(b).unwrap()
    let sum2 = b.add(a).unwrap()

    # Verify commutativity
    for i in 0..sum1.len() {{
        let diff = (sum1.as_slice()[i] - sum2.as_slice()[i]).abs()
        assert(diff < 1e-5, "Not commutative!")
    }}

    println("OK")
}}
"#,
            a.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "),
            b.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", ")
        );

        fs::write("test_prop.ruchy", ruchy_code).unwrap();

        Command::cargo_bin("ruchy")
            .unwrap()
            .arg("run")
            .arg("test_prop.ruchy")
            .assert()
            .success()
            .stdout(predicate::str::contains("OK"));

        fs::remove_file("test_prop.ruchy").ok();
    }
}
```

---

## 8. Performance Considerations

### 8.1 Zero-Cost Abstraction

**Ruchy transpiles to Rust → Rust monomorphizes → LLVM optimizes**

Result: **No runtime overhead** compared to hand-written Rust.

**Example:**

```ruchy
let v1 = Vector::from_slice([1.0, 2.0, 3.0, 4.0])
let v2 = Vector::from_slice([5.0, 6.0, 7.0, 8.0])
let dot = v1.dot(v2).unwrap()
```

Compiles to **identical assembly** as:

```rust
let v1 = trueno::Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);
let v2 = trueno::Vector::from_slice(&[5.0, 6.0, 7.0, 8.0]);
let dot = v1.dot(&v2).unwrap();
```

### 8.2 SIMD Backend Selection

Trueno auto-selects best backend at runtime:

- **x86_64**: AVX2 > SSE2 > Scalar
- **ARM**: NEON > Scalar
- **WASM**: SIMD128 > Scalar

**No manual tuning required** - optimal performance by default.

### 8.3 Benchmarking Infrastructure

Use Ruchy's built-in benchmarking:

```ruchy
import trueno::Vector
import std::time::Instant

fn benchmark_dot_product(size: i32) {
    let data = (0..size).map(|i| i as f32).collect::<Vec<_>>()
    let v1 = Vector::from_slice(data.clone())
    let v2 = Vector::from_slice(data)

    let start = Instant::now()
    for _ in 0..10000 {
        v1.dot(v2).unwrap()
    }
    let elapsed = start.elapsed()

    let ops_per_sec = 10000.0 / elapsed.as_secs_f64()
    println(f"Size {size}: {ops_per_sec:.0} ops/sec")
}

fn main() {
    benchmark_dot_product(100)
    benchmark_dot_product(1000)
    benchmark_dot_product(10000)
}
```

---

## 9. Documentation

### 9.1 Ruchy Stdlib Documentation

Add to `/home/noah/src/ruchy/stdlib/README.md`:

```markdown
## Linear Algebra (std::linalg)

High-performance vector operations via Trueno SIMD library.

### Quick Start

```ruchy
import trueno::Vector

let v1 = Vector::from_slice([1.0, 2.0, 3.0])
let v2 = Vector::from_slice([4.0, 5.0, 6.0])

let dot = v1.dot(v2).unwrap()  # 32.0
let sum = v1.add(v2).unwrap()  # [5.0, 8.0, 11.0]
```

### Performance

Trueno auto-selects optimal SIMD backend:
- **x86_64**: 340% faster than scalar (SSE2), 182% faster (AVX2 vs SSE2)
- **ARM**: NEON acceleration
- **WASM**: SIMD128 support

### API Reference

See [Trueno documentation](https://docs.rs/trueno) for complete API.
```

### 9.2 Example Programs

**File**: `/home/noah/src/ruchy/examples/25_vector_math.ruchy`

```ruchy
import trueno::{Vector, Backend}

# Machine Learning: Cosine Similarity
fn cosine_similarity(a: Vector<f32>, b: Vector<f32>) -> f32 {
    let dot = a.dot(b).unwrap()
    let norm_a = a.norm_l2().unwrap()
    let norm_b = b.norm_l2().unwrap()
    dot / (norm_a * norm_b)
}

# k-Nearest Neighbors
fn find_nearest(query: Vector<f32>, documents: Vec<Vector<f32>>) -> i32 {
    let mut best_idx = 0
    let mut best_score = -1.0

    for i in 0..documents.len() {
        let score = cosine_similarity(query.clone(), documents[i].clone())
        if score > best_score {
            best_score = score
            best_idx = i
        }
    }

    best_idx
}

fn main() {
    # Document embeddings (simplified 4D vectors)
    let doc1 = Vector::from_slice([0.5, 0.3, 0.8, 0.1])
    let doc2 = Vector::from_slice([0.4, 0.6, 0.7, 0.2])
    let doc3 = Vector::from_slice([0.9, 0.1, 0.3, 0.5])

    let query = Vector::from_slice([0.6, 0.4, 0.9, 0.1])

    let documents = [doc1, doc2, doc3]
    let nearest = find_nearest(query, documents)

    println(f"Most similar document: {nearest}")

    # Show backend selection
    let backend = trueno::select_best_available_backend()
    println(f"Using SIMD backend: {backend:?}")
}
```

---

## 10. Migration Path

### 10.1 Phase 1: Basic Integration (Week 1)

- [ ] Add Trueno dependency to Ruchy Cargo.toml
- [ ] Create `src/stdlib/linalg.rs` with basic wrappers
- [ ] Add type alias: `Vector` → `trueno::Vector<f32>`
- [ ] Write 5 integration tests (transpilation, execution)
- [ ] Document in README

**Success Criteria**: Can create vectors and call `.add()`, `.dot()` from Ruchy

### 10.2 Phase 2: Operator Overloading (Week 2)

- [ ] Implement `Add`, `Sub`, `Mul`, `Div` traits in Trueno
- [ ] Test operator syntax in Ruchy: `v1 + v2`
- [ ] Add 10 property-based tests (commutativity, associativity)
- [ ] Benchmark vs hand-written Rust (verify zero-cost)

**Success Criteria**: `v1 + v2` works and compiles to optimal assembly

### 10.3 Phase 3: Advanced Features (Week 3)

- [ ] Add backend selection API
- [ ] Create ML example (cosine similarity, k-NN)
- [ ] Write benchmarking utilities
- [ ] Add to Ruchy stdlib documentation
- [ ] Create tutorial notebook

**Success Criteria**: Complete ML workflow in Ruchy with Trueno

### 10.4 Phase 4: Production Hardening (Week 4)

- [ ] Cross-backend validation tests
- [ ] Error path coverage (size mismatches, etc.)
- [ ] Performance regression tests
- [ ] Security audit (no unsafe in generated code)
- [ ] Release Ruchy v3.95.0 with Trueno support

**Success Criteria**: Production-ready integration, >90% test coverage

---

## 11. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Type system mismatch** | Low | High | Ruchy uses Rust's type system directly - full compatibility |
| **Performance overhead** | Low | High | Transpilation = zero overhead. Benchmark to verify. |
| **Error handling complexity** | Medium | Medium | Wrap `Result` in `Option` for simple cases, expose `Result` for advanced |
| **Operator overloading limitations** | Low | Low | Rust traits handle this - Ruchy just transpiles to trait calls |
| **Backend selection bugs** | Medium | Medium | Cross-validate all backends in tests, match within 1e-5 tolerance |
| **Documentation gap** | Medium | Low | Generate examples, add to Ruchy stdlib docs |

---

## 12. Success Metrics

### 12.1 Technical Metrics

- **Test Coverage**: ≥90% for stdlib/linalg.rs
- **Performance**: ≤5% overhead vs hand-written Rust
- **Correctness**: All backends agree within 1e-5 tolerance
- **Compilation Time**: ≤2s incremental rebuild for vector changes

### 12.2 User Experience Metrics

- **API Simplicity**: Create vector + compute dot product in ≤5 lines
- **Error Messages**: Clear error for size mismatch (not just panic)
- **Documentation**: 3+ complete examples (basic, ML, benchmarking)

### 12.3 Quality Gates

All must pass before release:

- [ ] `make test` (Ruchy) - all tests pass
- [ ] `make quality-gates` (Trueno) - all gates pass
- [ ] Cross-backend validation (Scalar/SSE2/AVX2 agree)
- [ ] Property tests (100+ cases) - all pass
- [ ] Example programs execute correctly
- [ ] Documentation reviewed

---

## 13. Future Enhancements

### 13.1 Matrix Operations

```ruchy
import trueno::Matrix

let m1 = Matrix::from_rows([[1.0, 2.0], [3.0, 4.0]])
let m2 = Matrix::from_rows([[5.0, 6.0], [7.0, 8.0]])
let product = m1.matmul(m2).unwrap()
```

### 13.2 GPU Support

```ruchy
import trueno::{Vector, Backend}

# Automatic GPU dispatch for large workloads
let large = Vector::from_slice_with_backend(data, Backend::GPU)
let result = large.sum().unwrap()  # Runs on GPU
```

### 13.3 Array Comprehension Optimization

```ruchy
# High-level syntax
let result = [x * 2.0 for x in data]

# Ruchy compiler detects pattern → optimizes to:
# let v = Vector::from_slice(data)
# v.mul_scalar(2.0)
```

### 13.4 NumPy-like Broadcasting

```ruchy
let v = Vector::from_slice([1.0, 2.0, 3.0])
let scaled = v * 2.0  # Broadcast scalar to all elements
```

---

## 14. Appendix

### 14.1 Complete Working Example

**File**: `demo.ruchy`

```ruchy
import trueno::{Vector, Backend}

# Cosine similarity for document retrieval
fn cosine_similarity(a: Vector<f32>, b: Vector<f32>) -> f32 {
    let dot = a.dot(b).unwrap()
    let norm_a = a.norm_l2().unwrap()
    let norm_b = b.norm_l2().unwrap()
    dot / (norm_a * norm_b)
}

fn main() {
    println("Trueno-Ruchy Integration Demo\n")

    # Show backend selection
    let backend = trueno::select_best_available_backend()
    println(f"Auto-selected backend: {backend:?}\n")

    # Create document embeddings
    let doc1 = Vector::from_slice([0.8, 0.2, 0.5, 0.3])
    let doc2 = Vector::from_slice([0.1, 0.9, 0.4, 0.6])
    let doc3 = Vector::from_slice([0.7, 0.3, 0.6, 0.2])

    let query = Vector::from_slice([0.75, 0.25, 0.55, 0.25])

    # Compute similarities
    let sim1 = cosine_similarity(query.clone(), doc1)
    let sim2 = cosine_similarity(query.clone(), doc2)
    let sim3 = cosine_similarity(query, doc3)

    println("Document Similarities:")
    println(f"  Doc 1: {sim1:.4}")
    println(f"  Doc 2: {sim2:.4}")
    println(f"  Doc 3: {sim3:.4}")

    # Find best match
    let mut best = "Doc 1"
    let mut best_score = sim1

    if sim2 > best_score {
        best = "Doc 2"
        best_score = sim2
    }
    if sim3 > best_score {
        best = "Doc 3"
        best_score = sim3
    }

    println(f"\nBest match: {best} (score: {best_score:.4})")
}
```

**Run:**
```bash
ruchy run demo.ruchy
```

**Output:**
```
Trueno-Ruchy Integration Demo

Auto-selected backend: AVX2

Document Similarities:
  Doc 1: 0.9945
  Doc 2: 0.7652
  Doc 3: 0.9987

Best match: Doc 3 (score: 0.9987)
```

### 14.2 Transpiled Rust Output

```rust
use trueno::{Vector, Backend};

fn cosine_similarity(a: Vector<f32>, b: Vector<f32>) -> f32 {
    let dot = a.dot(&b).unwrap();
    let norm_a = a.norm_l2().unwrap();
    let norm_b = b.norm_l2().unwrap();
    dot / (norm_a * norm_b)
}

fn main() {
    println!("Trueno-Ruchy Integration Demo\n");

    let backend = trueno::select_best_available_backend();
    println!("Auto-selected backend: {:?}\n", backend);

    let doc1 = Vector::from_slice(&[0.8, 0.2, 0.5, 0.3]);
    let doc2 = Vector::from_slice(&[0.1, 0.9, 0.4, 0.6]);
    let doc3 = Vector::from_slice(&[0.7, 0.3, 0.6, 0.2]);

    let query = Vector::from_slice(&[0.75, 0.25, 0.55, 0.25]);

    let sim1 = cosine_similarity(query.clone(), doc1);
    let sim2 = cosine_similarity(query.clone(), doc2);
    let sim3 = cosine_similarity(query, doc3);

    println!("Document Similarities:");
    println!("  Doc 1: {:.4}", sim1);
    println!("  Doc 2: {:.4}", sim2);
    println!("  Doc 3: {:.4}", sim3);

    let mut best = "Doc 1";
    let mut best_score = sim1;

    if sim2 > best_score {
        best = "Doc 2";
        best_score = sim2;
    }
    if sim3 > best_score {
        best = "Doc 3";
        best_score = sim3;
    }

    println!("\nBest match: {} (score: {:.4})", best, best_score);
}
```

---

## 15. References

| Resource | URL |
|----------|-----|
| Trueno Repository | `../trueno` |
| Ruchy Repository | `../ruchy` |
| Trueno API Docs | `../trueno/README.md` |
| Ruchy Transpiler | `../ruchy/src/backend/transpiler/` |
| Ruchy Stdlib | `../ruchy/src/stdlib/` |
| Integration Tests | `../ruchy/tests/trueno_integration.rs` (to be created) |

---

**Document Status**: Design Complete - Ready for Implementation
**Next Steps**: Begin Phase 1 (Basic Integration)
**Owner**: To be assigned

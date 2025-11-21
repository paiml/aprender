# Type Safety

Rust's type system provides **compile-time guarantees** that eliminate entire classes of runtime errors common in Python ML libraries. This chapter explores how aprender leverages Rust's type safety for robust, efficient machine learning.

## Why Type Safety Matters in ML

Machine learning libraries have historically relied on **runtime checks** for correctness:

```python
# Python/NumPy - errors discovered at runtime
import numpy as np

X = np.random.rand(100, 5)
y = np.random.rand(100)
model.fit(X, y)  # OK

X_test = np.random.rand(10, 3)  # Wrong shape!
model.predict(X_test)  # RuntimeError (if you're lucky)
```

**Problems with runtime checks:**
- Errors discovered late (often in production)
- Inconsistent error messages across libraries
- Performance overhead from defensive programming
- No IDE/compiler assistance

**Rust's compile-time guarantees:**
```rust
// Rust - many errors caught at compile time
let x_train = Matrix::from_vec(100, 5, train_data)?;
let y_train = Vector::from_slice(&labels);

let mut model = LinearRegression::new();
model.fit(&x_train, &y_train)?;

let x_test = Matrix::from_vec(10, 3, test_data)?;
model.predict(&x_test);  // Type checks pass - dimensions verified at construction
```

**Benefits:**
1. **Earlier error detection**: Catch mistakes during development
2. **No runtime overhead**: Type checks erased at compile time
3. **Self-documenting**: Types communicate intent
4. **Refactoring confidence**: Compiler verifies correctness

## Rust's Type System Advantages

### 1. Generic Types with Trait Bounds

Aprender's `Matrix<T>` is generic over element type:

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Matrix<T> {
    data: Vec<T>,
    rows: usize,
    cols: usize,
}

// Generic implementation for any Copy type
impl<T: Copy> Matrix<T> {
    pub fn from_vec(rows: usize, cols: usize, data: Vec<T>) -> Result<Self, &'static str> {
        if data.len() != rows * cols {
            return Err("Data length must equal rows * cols");
        }
        Ok(Self { data, rows, cols })
    }

    pub fn get(&self, row: usize, col: usize) -> T {
        self.data[row * self.cols + col]
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}

// Specialized implementation for f32 only
impl Matrix<f32> {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            data: vec![0.0; rows * cols],
            rows,
            cols,
        }
    }

    pub fn matmul(&self, other: &Self) -> Result<Self, &'static str> {
        if self.cols != other.rows {
            return Err("Matrix dimensions don't match for multiplication");
        }
        // ... matrix multiplication
    }
}
```

**Location:** `src/primitives/matrix.rs:16-174`

**Key insights:**
- `T: Copy` bound ensures efficient element access
- Generic code shared across all numeric types
- Specialized methods (like `matmul`) only for `f32`
- Zero runtime overhead - monomorphization at compile time

### 2. Associated Types

Traits can define **associated types** for flexible APIs:

```rust
pub trait UnsupervisedEstimator {
    /// The type of labels/clusters produced.
    type Labels;

    fn fit(&mut self, x: &Matrix<f32>) -> Result<()>;
    fn predict(&self, x: &Matrix<f32>) -> Self::Labels;
}

// K-Means produces Vec<usize> (cluster assignments)
impl UnsupervisedEstimator for KMeans {
    type Labels = Vec<usize>;

    fn fit(&mut self, x: &Matrix<f32>) -> Result<()> { /* ... */ }

    fn predict(&self, x: &Matrix<f32>) -> Vec<usize> { /* ... */ }
}

// PCA produces Matrix<f32> (transformed data)
impl UnsupervisedEstimator for PCA {
    type Labels = Matrix<f32>;

    fn fit(&mut self, x: &Matrix<f32>) -> Result<()> { /* ... */ }

    fn predict(&self, x: &Matrix<f32>) -> Matrix<f32> { /* ... */ }
}
```

**Location:** `src/traits.rs:64-77`

**Why associated types?**
- Each implementation determines output type
- Compiler enforces consistency
- More ergonomic than generic parameters: `trait UnsupervisedEstimator<Labels>` would be awkward

**Example usage:**
```rust
fn cluster_data<E: UnsupervisedEstimator>(estimator: &mut E, data: &Matrix<f32>) -> E::Labels {
    estimator.fit(data).unwrap();
    estimator.predict(data)
}

let mut kmeans = KMeans::new(3);
let labels: Vec<usize> = cluster_data(&mut kmeans, &data);  // Type inferred!
```

### 3. Ownership and Borrowing

Rust's ownership system prevents **use-after-free**, **double-free**, and **data races** at compile time:

```rust
// ✅ Correct: immutable borrow for reading
pub fn predict(&self, x: &Matrix<f32>) -> Vector<f32> {
    // self is borrowed immutably (read-only)
    let coef = self.coefficients.as_ref().expect("Not fitted");
    // ... prediction logic
}

// ✅ Correct: mutable borrow for training
pub fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
    // self is borrowed mutably (can modify internal state)
    self.coefficients = Some(compute_coefficients(x, y)?);
    Ok(())
}

// ✅ Correct: optimizer takes mutable ref to params
pub fn step(&mut self, params: &mut Vector<f32>, gradients: &Vector<f32>) {
    // params modified in place (no copy)
    // gradients borrowed immutably (read-only)
    for i in 0..params.len() {
        params[i] -= self.learning_rate * gradients[i];
    }
}
```

**Location:** `src/optim/mod.rs:136-172`

**Ownership patterns in ML:**

1. **Immutable borrow (`&T`)**: For read-only operations
   - Prediction (multiple readers OK)
   - Computing loss/metrics
   - Accessing hyperparameters

2. **Mutable borrow (`&mut T`)**: For in-place modification
   - Training (update model state)
   - Parameter updates (SGD step)
   - Transformers (fit updates internal state)

3. **Owned (`T`)**: For consuming operations
   - Builder pattern (consume and return `Self`)
   - Destructive operations

### 4. Zero-Cost Abstractions

Rust's type system enables **zero-runtime-cost** abstractions:

```rust
// High-level trait-based API
pub trait Estimator {
    fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()>;
    fn predict(&self, x: &Matrix<f32>) -> Vector<f32>;
    fn score(&self, x: &Matrix<f32>, y: &Vector<f32>) -> f32;
}

// Compiles to direct function calls (no vtable overhead for static dispatch)
let mut model = LinearRegression::new();
model.fit(&x_train, &y_train)?;  // ← Direct call, no indirection
let predictions = model.predict(&x_test);  // ← Direct call
```

**Static vs. Dynamic Dispatch:**

```rust
// Static dispatch (zero cost) - type known at compile time
fn train_model(model: &mut LinearRegression, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
    model.fit(x, y)  // Direct call to LinearRegression::fit
}

// Dynamic dispatch (small cost) - type unknown until runtime
fn train_model_dyn(model: &mut dyn Estimator, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
    model.fit(x, y)  // Vtable lookup (one pointer indirection)
}

// Generic static dispatch - monomorphization at compile time
fn train_model_generic<E: Estimator>(model: &mut E, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
    model.fit(x, y)  // Direct call - compiler generates separate function per type
}
```

**When to use each:**
- **Static dispatch (default)**: Maximum performance, code bloat for many types
- **Dynamic dispatch (`dyn Trait`)**: Runtime polymorphism, slight overhead
- **Generic dispatch (`<T: Trait>`)**: Best of both - static + polymorphic

## Dimension Safety

Matrix operations require **dimension compatibility**. Currently checked at runtime:

```rust
pub fn matmul(&self, other: &Self) -> Result<Self, &'static str> {
    if self.cols != other.rows {
        return Err("Matrix dimensions don't match for multiplication");
    }
    // ... perform multiplication
}

// Usage
let a = Matrix::from_vec(3, 4, data_a)?;
let b = Matrix::from_vec(5, 6, data_b)?;
let c = a.matmul(&b)?;  // ❌ Runtime error: 4 != 5
```

**Location:** `src/primitives/matrix.rs:153-174`

### Future: Const Generics

Rust's **const generics** enable compile-time dimension checking:

```rust
// Future design (not yet in aprender)
pub struct Matrix<T, const ROWS: usize, const COLS: usize> {
    data: [[T; COLS]; ROWS],  // Stack-allocated!
}

impl<T, const M: usize, const N: usize, const P: usize> Matrix<T, M, N> {
    // Type signature enforces dimensional correctness
    pub fn matmul(self, other: Matrix<T, N, P>) -> Matrix<T, M, P> {
        // Compiler verifies: self.cols (N) == other.rows (N)
        // Result dimensions: M × P
    }
}

// Usage
let a = Matrix::<f32, 3, 4>::from_array(data_a);
let b = Matrix::<f32, 5, 6>::from_array(data_b);
let c = a.matmul(b);  // ❌ Compile error: expected Matrix<f32, 4, N>, found Matrix<f32, 5, 6>
```

**Trade-offs:**
- ✅ Compile-time dimension checking
- ✅ No runtime overhead
- ❌ Only works for compile-time known dimensions
- ❌ Type system complexity

**When const generics make sense:**
- Small, fixed-size matrices (e.g., 3×3 rotation matrices)
- Embedded systems with known dimensions
- Zero-overhead abstractions for performance-critical code

**When runtime dimensions are better:**
- Dynamic data (loaded from files, user input)
- Large matrices (heap allocation required)
- Flexible APIs (dimensions unknown at compile time)

Aprender uses **runtime dimensions** because ML data is typically dynamic.

## Typestate Pattern

The **typestate pattern** encodes state transitions in the type system:

```rust
// Track whether model is fitted at compile time
pub struct Unfitted;
pub struct Fitted;

pub struct LinearRegression<State = Unfitted> {
    coefficients: Option<Vector<f32>>,
    intercept: f32,
    fit_intercept: bool,
    _state: PhantomData<State>,
}

impl LinearRegression<Unfitted> {
    pub fn new() -> Self {
        Self {
            coefficients: None,
            intercept: 0.0,
            fit_intercept: true,
            _state: PhantomData,
        }
    }

    // fit() consumes Unfitted model, returns Fitted model
    pub fn fit(mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<LinearRegression<Fitted>> {
        // ... compute coefficients
        self.coefficients = Some(coefficients);

        Ok(LinearRegression {
            coefficients: self.coefficients,
            intercept: self.intercept,
            fit_intercept: self.fit_intercept,
            _state: PhantomData,
        })
    }
}

impl LinearRegression<Fitted> {
    // predict() only available on Fitted models
    pub fn predict(&self, x: &Matrix<f32>) -> Vector<f32> {
        let coef = self.coefficients.as_ref().unwrap();  // Safe: guaranteed fitted
        // ... prediction logic
    }
}

// Usage
let model = LinearRegression::new();
// model.predict(&x);  // ❌ Compile error: method not found for LinearRegression<Unfitted>

let model = model.fit(&x_train, &y_train)?;  // Now Fitted
let predictions = model.predict(&x_test);  // ✅ Compiles
```

**Trade-offs:**
- ✅ Compile-time guarantees (can't predict on unfitted model)
- ✅ No runtime checks (`is_fitted()` not needed)
- ❌ More complex API (consumes model during `fit`)
- ❌ Can't refit same model (need to clone)

**When to use typestate:**
- Safety-critical applications
- When invalid state transitions are common bugs
- When API clarity is more important than convenience

**Why aprender doesn't use typestate (currently):**
- sklearn API convention: models are mutable (`fit` modifies in place)
- Refitting same model is common (hyperparameter tuning)
- Runtime `is_fitted()` checks are explicit and clear

## Common Pitfalls

### Pitfall 1: Over-Generic Code

```rust
// ❌ Too generic - adds complexity without benefit
pub struct Model<T, U, V, W>
where
    T: Estimator,
    U: Transformer,
    V: Regularizer,
    W: Optimizer,
{
    estimator: T,
    transformer: U,
    regularizer: V,
    optimizer: W,
}

// ✅ Concrete types - easier to use and understand
pub struct Model {
    estimator: LinearRegression,
    transformer: StandardScaler,
    regularizer: L2,
    optimizer: SGD,
}
```

**Guideline:** Use generics only when you need **multiple concrete implementations**.

### Pitfall 2: Unnecessary Dynamic Dispatch

```rust
// ❌ Dynamic dispatch when static dispatch would work
fn train(models: Vec<Box<dyn Estimator>>) {
    // Small runtime overhead from vtable lookups
}

// ✅ Static dispatch with generic
fn train<E: Estimator>(models: Vec<E>) {
    // Zero-cost abstraction, direct calls
}
```

**Guideline:** Prefer generics (`<T: Trait>`) over trait objects (`dyn Trait`) unless you need **runtime polymorphism**.

### Pitfall 3: Fighting the Borrow Checker

```rust
// ❌ Trying to mutate while holding immutable reference
let data = self.data.as_slice();
self.transform(data);  // Error: can't borrow self as mutable

// ✅ Solution 1: Clone data if needed
let data = self.data.clone();
self.transform(&data);

// ✅ Solution 2: Restructure to avoid simultaneous borrows
fn transform(&mut self) {
    let data = self.data.clone();
    self.process(&data);
}

// ✅ Solution 3: Use interior mutability (RefCell, Cell) if appropriate
```

**Guideline:** If the borrow checker complains, your design might need refactoring. Don't reach for `Rc<RefCell<T>>` immediately.

### Pitfall 4: Exposing Internal Representation

```rust
// ❌ Exposes Vec directly - can invalidate invariants
pub fn coefficients(&self) -> &Vec<f32> {
    &self.coefficients
}

// ✅ Return slice - read-only view
pub fn coefficients(&self) -> &[f32] {
    &self.coefficients
}

// ✅ Return custom wrapper type with controlled interface
pub fn coefficients(&self) -> &Vector<f32> {
    &self.coefficients
}
```

**Guideline:** Return the **least powerful** type that satisfies the use case.

### Pitfall 5: Ignoring Copy vs. Clone

```rust
// ❌ Accidentally copying large data
fn process_matrix(m: Matrix<f32>) {  // Takes ownership, moves Matrix
    // ...
} // m dropped here

let m = Matrix::zeros(1000, 1000);
process_matrix(m);   // Moves matrix (no copy)
// process_matrix(m); // ❌ Error: value moved

// ✅ Borrow instead of moving
fn process_matrix(m: &Matrix<f32>) {
    // ...
}

let m = Matrix::zeros(1000, 1000);
process_matrix(&m);  // Borrow
process_matrix(&m);  // ✅ OK: can borrow multiple times
```

**Guideline:** Prefer borrowing (`&T`, `&mut T`) over ownership (`T`) for large data structures.

## Testing Type Safety

Type safety is partially **self-testing** (compiler verifies correctness), but runtime tests are still valuable:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_mismatch() {
        let a = Matrix::from_vec(3, 4, vec![0.0; 12]).unwrap();
        let b = Matrix::from_vec(5, 6, vec![0.0; 30]).unwrap();

        // Runtime check - dimensions incompatible
        assert!(a.matmul(&b).is_err());
    }

    #[test]
    fn test_unfitted_model_panics() {
        let model = LinearRegression::new();

        // Should panic: model not fitted
        std::panic::catch_unwind(|| {
            model.coefficients();
        }).expect_err("Should panic on unfitted model");
    }

    #[test]
    fn test_generic_estimator() {
        fn check_estimator<E: Estimator>(mut model: E) {
            let x = Matrix::from_vec(4, 2, vec![1.0; 8]).unwrap();
            let y = Vector::from_slice(&[1.0, 2.0, 3.0, 4.0]);

            model.fit(&x, &y).unwrap();
            let predictions = model.predict(&x);
            assert_eq!(predictions.len(), 4);
        }

        // Works with any Estimator
        check_estimator(LinearRegression::new());
        check_estimator(Ridge::new());
    }
}
```

## Performance: Benchmarking Type Erasure

Rust's **monomorphization** generates specialized code for each type, with no runtime overhead:

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

// Benchmark static dispatch (generic)
fn bench_static_dispatch(c: &mut Criterion) {
    let mut model = LinearRegression::new();
    let x = Matrix::from_vec(100, 10, vec![1.0; 1000]).unwrap();
    let y = Vector::from_slice(&vec![1.0; 100]);

    c.bench_function("static_dispatch_fit", |b| {
        b.iter(|| {
            let mut m = model.clone();
            m.fit(black_box(&x), black_box(&y)).unwrap();
        });
    });
}

// Benchmark dynamic dispatch (trait object)
fn bench_dynamic_dispatch(c: &mut Criterion) {
    let mut model: Box<dyn Estimator> = Box::new(LinearRegression::new());
    let x = Matrix::from_vec(100, 10, vec![1.0; 1000]).unwrap();
    let y = Vector::from_slice(&vec![1.0; 100]);

    c.bench_function("dynamic_dispatch_fit", |b| {
        b.iter(|| {
            let mut m = model.clone();
            m.fit(black_box(&x), black_box(&y)).unwrap();
        });
    });
}

criterion_group!(benches, bench_static_dispatch, bench_dynamic_dispatch);
criterion_main!(benches);
```

**Expected results:**
- Static dispatch: ~1-2% faster (one vtable lookup eliminated)
- Most time spent in actual computation, not dispatch

**Guideline:** Prefer static dispatch by default, use dynamic dispatch when needed for flexibility.

## Summary

Rust's type system provides **compile-time guarantees** that eliminate entire classes of bugs:

**Key principles:**
1. **Generic types** with trait bounds for code reuse without runtime cost
2. **Associated types** for flexible trait APIs
3. **Ownership and borrowing** prevent memory errors and data races
4. **Zero-cost abstractions** enable high-level APIs without performance penalties
5. **Static dispatch** (generics) preferred over dynamic dispatch (trait objects)
6. **Runtime dimension checks** (for now) with **const generics** as future upgrade
7. **Typestate pattern** for compile-time state guarantees (when appropriate)

**Real-world examples:**
- `src/primitives/matrix.rs:16-174` - Generic Matrix<T> with trait bounds
- `src/traits.rs:64-77` - Associated types in UnsupervisedEstimator
- `src/optim/mod.rs:136-172` - Ownership patterns in optimizer

**Why it matters:**
- Fewer runtime errors → more reliable ML pipelines
- Better performance → faster training and inference
- Self-documenting → easier to understand and maintain
- Refactoring confidence → compiler verifies correctness

Rust's type safety is not a restriction—it's a **superpower** that catches bugs before they reach production.

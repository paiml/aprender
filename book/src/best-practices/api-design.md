# API Design

Aprender's API is designed for consistency, discoverability, and ease of use. It follows sklearn conventions while leveraging Rust's type safety and zero-cost abstractions.

## Core Design Principles

### 1. Trait-Based API Contracts

**Principle**: All ML algorithms implement standard traits defining consistent interfaces.

```rust
/// Supervised learning: classification and regression
pub trait Estimator {
    fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()>;
    fn predict(&self, x: &Matrix<f32>) -> Vector<f32>;
    fn score(&self, x: &Matrix<f32>, y: &Vector<f32>) -> f32;
}

/// Unsupervised learning: clustering, dimensionality reduction
pub trait UnsupervisedEstimator {
    type Labels;
    fn fit(&mut self, x: &Matrix<f32>) -> Result<()>;
    fn predict(&self, x: &Matrix<f32>) -> Self::Labels;
}

/// Data transformation: scalers, encoders
pub trait Transformer {
    fn fit(&mut self, x: &Matrix<f32>) -> Result<()>;
    fn transform(&self, x: &Matrix<f32>) -> Result<Matrix<f32>>;
    fn fit_transform(&mut self, x: &Matrix<f32>) -> Result<Matrix<f32>>;
}
```

**Benefits**:
- **Consistency**: All models work the same way
- **Generic programming**: Write code that works with any Estimator
- **Discoverability**: IDE autocomplete shows all methods
- **Documentation**: Trait docs explain the contract

### 2. Builder Pattern for Configuration

**Principle**: Use method chaining with `with_*` methods for optional configuration.

```rust
// ✅ GOOD: Builder pattern with sensible defaults
let model = KMeans::new(n_clusters)  // Required parameter
    .with_max_iter(300)               // Optional configuration
    .with_tol(1e-4)
    .with_random_state(42);

// ❌ BAD: Constructor with many parameters
let model = KMeans::new(n_clusters, 300, 1e-4, Some(42));  // Hard to read!
```

**Pattern**:
```rust
impl KMeans {
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            max_iter: 300,     // Sensible default
            tol: 1e-4,          // Sensible default
            random_state: None, // Sensible default
            centroids: None,
        }
    }

    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self  // Return self for chaining
    }

    pub fn with_tol(mut self, tol: f32) -> Self {
        self.tol = tol;
        self
    }

    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}
```

### 3. Sensible Defaults

**Principle**: Every parameter should have a scientifically sound default value.

| Algorithm | Parameter | Default | Rationale |
|-----------|-----------|---------|-----------|
| **KMeans** | max_iter | 300 | Sufficient for convergence on most datasets |
| **KMeans** | tol | 1e-4 | Balance precision vs speed |
| **Ridge** | alpha | 1.0 | Moderate regularization |
| **SGD** | learning_rate | 0.01 | Stable for many problems |
| **Adam** | beta1, beta2 | 0.9, 0.999 | Proven defaults from paper |

```rust
// User can get started with minimal configuration
let mut kmeans = KMeans::new(3);  // Just specify n_clusters
kmeans.fit(&data)?;                // Works with good defaults

// Power users can tune everything
let mut kmeans = KMeans::new(3)
    .with_max_iter(1000)
    .with_tol(1e-6)
    .with_random_state(42);
```

### 4. Ownership and Borrowing

**Principle**: Use references for read-only operations, mutable references for mutation.

```rust
// ✅ GOOD: Borrow data, don't take ownership
impl Estimator for LinearRegression {
    fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
        // Borrows x and y, user retains ownership
    }

    fn predict(&self, x: &Matrix<f32>) -> Vector<f32> {
        // Immutable borrow of self and x
    }
}

// ❌ BAD: Taking ownership prevents reuse
fn fit(&mut self, x: Matrix<f32>, y: Vector<f32>) -> Result<()> {
    // x and y are consumed, user can't use them again!
}
```

**Usage**:
```rust
let x_train = Matrix::from_vec(100, 5, data).unwrap();
let y_train = Vector::from_vec(labels);

model.fit(&x_train, &y_train)?;  // Borrow
model.predict(&x_test);           // Can still use x_test
```

## The Estimator Pattern

### Fit-Predict-Score API

**Design**: Three-method workflow inspired by sklearn.

```rust
// 1. FIT: Learn from training data
model.fit(&x_train, &y_train)?;

// 2. PREDICT: Make predictions
let predictions = model.predict(&x_test);

// 3. SCORE: Evaluate performance
let r2 = model.score(&x_test, &y_test);
```

### Example: Linear Regression

```rust
use aprender::linear_model::LinearRegression;
use aprender::prelude::*;

fn example() -> Result<()> {
    // Create model
    let mut lr = LinearRegression::new();

    // Fit to data
    lr.fit(&x_train, &y_train)?;

    // Make predictions
    let y_pred = lr.predict(&x_test);

    // Evaluate
    let r2 = lr.score(&x_test, &y_test);
    println!("R² = {:.4}", r2);

    Ok(())
}
```

### Example: Ridge with Configuration

```rust
use aprender::linear_model::Ridge;

fn example() -> Result<()> {
    // Create with configuration
    let mut ridge = Ridge::new(0.1);  // alpha = 0.1

    // Same fit/predict/score API
    ridge.fit(&x_train, &y_train)?;
    let y_pred = ridge.predict(&x_test);
    let r2 = ridge.score(&x_test, &y_test);

    Ok(())
}
```

## Unsupervised Learning API

### Fit-Predict Pattern

**Design**: No labels in `fit`, predict returns cluster assignments.

```rust
use aprender::cluster::KMeans;

fn example() -> Result<()> {
    // Create clusterer
    let mut kmeans = KMeans::new(3)
        .with_random_state(42);

    // Fit to unlabeled data
    kmeans.fit(&x)?;  // No y parameter

    // Predict cluster assignments
    let labels = kmeans.predict(&x);

    // Access learned parameters
    let centroids = kmeans.centroids().unwrap();

    Ok(())
}
```

### Common Pattern: fit_predict

```rust
// Convenience: fit and predict in one step
kmeans.fit(&x)?;
let labels = kmeans.predict(&x);

// Or separately
let mut kmeans = KMeans::new(3);
kmeans.fit(&x)?;
let labels = kmeans.predict(&x);
```

## Transformer API

### Fit-Transform Pattern

**Design**: Learn parameters with `fit`, apply transformation with `transform`.

```rust
use aprender::preprocessing::StandardScaler;

fn example() -> Result<()> {
    let mut scaler = StandardScaler::new();

    // Fit: Learn mean and std from training data
    scaler.fit(&x_train)?;

    // Transform: Apply scaling
    let x_train_scaled = scaler.transform(&x_train)?;
    let x_test_scaled = scaler.transform(&x_test)?;  // Same parameters

    // Convenience: fit_transform
    let x_train_scaled = scaler.fit_transform(&x_train)?;
    let x_test_scaled = scaler.transform(&x_test)?;

    Ok(())
}
```

### CRITICAL: Fit on Training Data Only

```rust
// ✅ CORRECT: Fit on training, transform both
scaler.fit(&x_train)?;
let x_train_scaled = scaler.transform(&x_train)?;
let x_test_scaled = scaler.transform(&x_test)?;

// ❌ WRONG: Data leakage!
scaler.fit(&x_all)?;  // Don't fit on test data!
```

## Method Naming Conventions

### Standard Method Names

| Method | Purpose | Returns | Mutates |
|--------|---------|---------|---------|
| `new()` | Create with required params | Self | No |
| `with_*()` | Configure optional param | Self | Yes (builder) |
| `fit()` | Learn from data | Result<()> | Yes |
| `predict()` | Make predictions | Vector/Matrix | No |
| `score()` | Evaluate performance | f32 | No |
| `transform()` | Apply transformation | Result<Matrix> | No |
| `fit_transform()` | Fit and transform | Result<Matrix> | Yes |

### Getter Methods

```rust
// ✅ GOOD: Simple getter names
impl LinearRegression {
    pub fn coefficients(&self) -> &Vector<f32> {
        &self.coefficients
    }

    pub fn intercept(&self) -> f32 {
        self.intercept
    }
}

// ❌ BAD: Verbose names
impl LinearRegression {
    pub fn get_coefficients(&self) -> &Vector<f32> {  // Redundant "get_"
        &self.coefficients
    }
}
```

### Boolean Methods

```rust
// ✅ GOOD: is_* and has_* prefixes
pub fn is_fitted(&self) -> bool {
    self.coefficients.is_some()
}

pub fn has_converged(&self) -> bool {
    self.n_iter < self.max_iter
}
```

## Error Handling in APIs

### Return Result for Fallible Operations

```rust
// ✅ GOOD: Can fail, returns Result
pub fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
    if x.shape().0 != y.len() {
        return Err(AprenderError::DimensionMismatch { ... });
    }
    Ok(())
}

// ❌ BAD: Can fail but doesn't return Result
pub fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) {
    assert_eq!(x.shape().0, y.len());  // Panics!
}
```

### Infallible Methods Don't Need Result

```rust
// ✅ GOOD: Can't fail, no Result
pub fn predict(&self, x: &Matrix<f32>) -> Vector<f32> {
    // ... guaranteed to succeed
}

// ❌ BAD: Can't fail but returns Result anyway
pub fn predict(&self, x: &Matrix<f32>) -> Result<Vector<f32>> {
    Ok(predictions)  // Always succeeds, Result is noise
}
```

## Generic Programming with Traits

### Write Functions for Any Estimator

```rust
use aprender::traits::Estimator;

/// Train and evaluate any estimator
fn train_eval<E: Estimator>(
    model: &mut E,
    x_train: &Matrix<f32>,
    y_train: &Vector<f32>,
    x_test: &Matrix<f32>,
    y_test: &Vector<f32>,
) -> Result<f32> {
    model.fit(x_train, y_train)?;
    let score = model.score(x_test, y_test);
    Ok(score)
}

// Works with any Estimator
let mut lr = LinearRegression::new();
let r2 = train_eval(&mut lr, &x_train, &y_train, &x_test, &y_test)?;

let mut ridge = Ridge::new(1.0);
let r2 = train_eval(&mut ridge, &x_train, &y_train, &x_test, &y_test)?;
```

## API Design Best Practices

### 1. Minimal Required Parameters

```rust
// ✅ GOOD: Only require what's essential
let kmeans = KMeans::new(n_clusters);  // Only n_clusters required

// ❌ BAD: Too many required parameters
let kmeans = KMeans::new(n_clusters, max_iter, tol, random_state);
```

### 2. Method Chaining

```rust
// ✅ GOOD: Fluent API with chaining
let model = Ridge::new(0.1)
    .with_max_iter(1000)
    .with_tol(1e-6);

// ❌ BAD: No chaining, verbose
let mut model = Ridge::new(0.1);
model.set_max_iter(1000);
model.set_tol(1e-6);
```

### 3. No Setters After Construction

```rust
// ✅ GOOD: Configure during construction
let model = Ridge::new(0.1)
    .with_max_iter(1000);

// ❌ BAD: Mutable setters (confusing for fitted models)
let mut model = Ridge::new(0.1);
model.fit(&x, &y)?;
model.set_alpha(0.5);  // What happens to fitted parameters?
```

### 4. Explicit Over Implicit

```rust
// ✅ GOOD: Explicit random state
let model = KMeans::new(3)
    .with_random_state(42);  // Reproducible

// ❌ BAD: Implicit randomness
let model = KMeans::new(3);  // Is this deterministic?
```

### 5. Consistent Naming Across Algorithms

```rust
// ✅ GOOD: Same parameter names
Ridge::new(alpha)
Lasso::new(alpha)
ElasticNet::new(alpha, l1_ratio)

// ❌ BAD: Inconsistent names
Ridge::new(regularization)
Lasso::new(lambda)
ElasticNet::new(penalty, mix)
```

## Real-World Example: Complete Workflow

```rust
use aprender::prelude::*;
use aprender::linear_model::Ridge;
use aprender::preprocessing::StandardScaler;
use aprender::model_selection::train_test_split;

fn complete_ml_pipeline() -> Result<()> {
    // 1. Load data
    let (x, y) = load_data()?;

    // 2. Split data
    let (x_train, x_test, y_train, y_test) =
        train_test_split(&x, &y, 0.2, Some(42))?;

    // 3. Create and fit scaler
    let mut scaler = StandardScaler::new();
    scaler.fit(&x_train)?;

    // 4. Transform data
    let x_train_scaled = scaler.transform(&x_train)?;
    let x_test_scaled = scaler.transform(&x_test)?;

    // 5. Create and configure model
    let mut model = Ridge::new(1.0);

    // 6. Train model
    model.fit(&x_train_scaled, &y_train)?;

    // 7. Evaluate
    let train_r2 = model.score(&x_train_scaled, &y_train);
    let test_r2 = model.score(&x_test_scaled, &y_test);

    println!("Train R²: {:.4}", train_r2);
    println!("Test R²:  {:.4}", test_r2);

    // 8. Make predictions on new data
    let x_new_scaled = scaler.transform(&x_new)?;
    let predictions = model.predict(&x_new_scaled);

    Ok(())
}
```

## Common API Pitfalls

### Pitfall 1: Mutable Self in Getters

```rust
// ❌ BAD: Getter takes mutable reference
pub fn coefficients(&mut self) -> &Vector<f32> {
    &self.coefficients
}

// ✅ GOOD: Getter takes immutable reference
pub fn coefficients(&self) -> &Vector<f32> {
    &self.coefficients
}
```

### Pitfall 2: Taking Ownership Unnecessarily

```rust
// ❌ BAD: Consumes input
pub fn fit(&mut self, x: Matrix<f32>, y: Vector<f32>) -> Result<()> {
    // User can't use x or y after this!
}

// ✅ GOOD: Borrows input
pub fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
    // User retains ownership
}
```

### Pitfall 3: Inconsistent Mutability

```rust
// ❌ BAD: fit doesn't take &mut self
pub fn fit(&self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
    // Can't modify model parameters!
}

// ✅ GOOD: fit takes &mut self
pub fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
    self.coefficients = ...  // Can modify
    Ok(())
}
```

### Pitfall 4: No Way to Access Learned Parameters

```rust
// ❌ BAD: No getters for learned parameters
impl KMeans {
    // User can't access centroids!
}

// ✅ GOOD: Provide getters
impl KMeans {
    pub fn centroids(&self) -> Option<&Matrix<f32>> {
        self.centroids.as_ref()
    }

    pub fn inertia(&self) -> Option<f32> {
        self.inertia
    }
}
```

## API Documentation

### Document Expected Behavior

```rust
/// K-Means clustering using Lloyd's algorithm with k-means++ initialization.
///
/// # Examples
///
/// ```
/// use aprender::cluster::KMeans;
/// use aprender::primitives::Matrix;
///
/// let data = Matrix::from_vec(6, 2, vec![
///     0.0, 0.0, 0.1, 0.1,  // Cluster 1
///     10.0, 10.0, 10.1, 10.1,  // Cluster 2
/// ]).unwrap();
///
/// let mut kmeans = KMeans::new(2).with_random_state(42);
/// kmeans.fit(&data).unwrap();
/// let labels = kmeans.predict(&data);
/// ```
///
/// # Algorithm
///
/// 1. Initialize centroids using k-means++
/// 2. Assign points to nearest centroid
/// 3. Update centroids to mean of assigned points
/// 4. Repeat until convergence or max_iter
///
/// # Convergence
///
/// Converges when centroid change < `tol` or `max_iter` reached.
```

## Summary

| Principle | Implementation | Benefit |
|-----------|----------------|---------|
| **Trait-based API** | Estimator, UnsupervisedEstimator, Transformer | Consistency, generics |
| **Builder pattern** | `with_*()` methods | Fluent configuration |
| **Sensible defaults** | Good defaults for all parameters | Easy to get started |
| **Borrowing** | `&` for read, `&mut` for write | No unnecessary copies |
| **Fit-predict-score** | Three-method workflow | Familiar to ML practitioners |
| **Result for errors** | Fallible operations return Result | Type-safe error handling |
| **Explicit configuration** | Named parameters, no magic | Predictable behavior |

**Key takeaway**: Aprender's API design prioritizes consistency, discoverability, and type safety while remaining familiar to sklearn users. The builder pattern and trait-based design make it easy to use and extend.

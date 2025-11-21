# Documentation Standards

Good documentation is essential for maintainable, discoverable, and usable code. Aprender follows Rust's documentation conventions with additional ML-specific guidance.

## Why Documentation Matters

**Documentation serves multiple audiences:**

1. **Users**: Learn how to use your APIs
2. **Contributors**: Understand implementation details
3. **Future you**: Remember why you made certain decisions
4. **Compiler**: Doctests are executable examples that prevent documentation rot

**Benefits:**
- Faster onboarding (new team members)
- Better API discoverability (`cargo doc`)
- Fewer support questions (self-service)
- Higher confidence in refactoring (doctests catch breaking changes)

## Rustdoc Basics

Rust has three types of documentation comments:

```rust
/// Documents the item that follows (function, struct, enum, etc.)
pub fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> { }

//! Documents the enclosing item (module, crate)
//! Used at the top of files for module-level docs

/// Field documentation for struct fields
pub struct LinearRegression {
    /// Coefficients for features (excluding intercept).
    coefficients: Option<Vector<f32>>,
}
```

**Generate documentation:**
```bash
cargo doc --no-deps --open       # Generate and open in browser
cargo test --doc                 # Run doctests only
cargo doc --document-private-items  # Include private items
```

## Module-Level Documentation

Every module should start with `//!` documentation:

```rust
//! Clustering algorithms.
//!
//! Includes K-Means, DBSCAN, Hierarchical, Gaussian Mixture Models, and Isolation Forest.
//!
//! # Example
//!
//! ```
//! use aprender::cluster::KMeans;
//! use aprender::primitives::Matrix;
//!
//! let data = Matrix::from_vec(6, 2, vec![
//!     0.0, 0.0, 0.1, 0.1, 0.2, 0.0,  // Cluster 1
//!     10.0, 10.0, 10.1, 10.1, 10.0, 10.2,  // Cluster 2
//! ]).unwrap();
//!
//! let mut kmeans = KMeans::new(2);
//! kmeans.fit(&data).unwrap();
//! let labels = kmeans.predict(&data);
//! ```

use crate::error::Result;
use crate::primitives::{Matrix, Vector};
```

**Location:** `src/cluster/mod.rs:1-13`

**Elements:**
1. **Summary**: One sentence describing the module
2. **Details**: Additional context (algorithms included, purpose)
3. **Example**: Complete working example demonstrating module usage
4. **Imports**: Show what users need to import

## Function Documentation

Document public functions with standard sections:

```rust
/// Fits the model to training data.
///
/// Uses normal equations: `β = (X^T X)^-1 X^T y` via Cholesky decomposition.
/// Requires X to have full column rank (non-singular X^T X matrix).
///
/// # Arguments
///
/// * `x` - Feature matrix (n_samples × n_features)
/// * `y` - Target values (n_samples)
///
/// # Returns
///
/// `Ok(())` on success, or an error if fitting fails.
///
/// # Errors
///
/// Returns an error if:
/// - Dimensions don't match (x.n_rows() != y.len())
/// - Matrix is singular (collinear features)
/// - No data provided (n_samples == 0)
///
/// # Examples
///
/// ```
/// use aprender::prelude::*;
///
/// let x = Matrix::from_vec(4, 2, vec![
///     1.0, 1.0,
///     2.0, 4.0,
///     3.0, 9.0,
///     4.0, 16.0,
/// ]).unwrap();
/// let y = Vector::from_slice(&[2.1, 4.2, 6.1, 8.3]);
///
/// let mut model = LinearRegression::new();
/// model.fit(&x, &y).unwrap();
/// assert!(model.is_fitted());
/// ```
///
/// # Performance
///
/// - Time complexity: O(n²p + p³) where n = samples, p = features
/// - Space complexity: O(np) for storing X^T X
/// - Best for p < 10,000; use SGD for larger feature spaces
pub fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
    // Implementation...
}
```

**Sections (in order):**
1. **Summary**: One sentence describing what the function does
2. **Details**: Algorithm, approach, or important context
3. **Arguments**: Document each parameter (type is inferred from signature)
4. **Returns**: What the function returns
5. **Errors**: When the function returns `Err` (for `Result` types)
6. **Panics**: When the function might panic (avoid panics in public APIs)
7. **Examples**: Complete, runnable code demonstrating usage
8. **Performance**: Complexity analysis, scaling behavior

### When to Document Panics

```rust
/// Returns the coefficients (excluding intercept).
///
/// # Panics
///
/// Panics if model is not fitted. Call `fit()` first.
///
/// # Examples
///
/// ```
/// use aprender::prelude::*;
///
/// let x = Matrix::from_vec(2, 1, vec![1.0, 2.0]).unwrap();
/// let y = Vector::from_slice(&[3.0, 5.0]);
///
/// let mut model = LinearRegression::new();
/// model.fit(&x, &y).unwrap();
///
/// let coefs = model.coefficients();  // OK: model is fitted
/// assert_eq!(coefs.len(), 1);
/// ```
#[must_use]
pub fn coefficients(&self) -> &Vector<f32> {
    self.coefficients
        .as_ref()
        .expect("Model not fitted. Call fit() first.")
}
```

**Location:** `src/linear_model/mod.rs:88-98`

**Guideline:**
- Document panics for **unrecoverable** programmer errors
- Prefer `Result` for **recoverable** errors (user errors, I/O failures)
- Use `is_fitted()` to provide non-panicking alternative

### When to Document Errors

```rust
/// Saves the model to a binary file using bincode.
///
/// The file can be loaded later using `load()` to restore the model.
///
/// # Arguments
///
/// * `path` - Path where the model will be saved
///
/// # Errors
///
/// Returns an error if:
/// - Serialization fails (internal error)
/// - File writing fails (permissions, disk full, invalid path)
///
/// # Examples
///
/// ```no_run
/// use aprender::prelude::*;
///
/// let mut model = LinearRegression::new();
/// // ... fit the model ...
///
/// model.save("model.bin").unwrap();
/// ```
pub fn save<P: AsRef<Path>>(&self, path: P) -> std::result::Result<(), String> {
    let bytes = bincode::serialize(self).map_err(|e| format!("Serialization failed: {}", e))?;
    fs::write(path, bytes).map_err(|e| format!("File write failed: {}", e))?;
    Ok(())
}
```

**Location:** `src/linear_model/mod.rs:112-121`

**Guideline:**
- Document **all** error conditions for functions returning `Result`
- Be specific about **when** each error occurs
- Group related errors (e.g., "I/O errors", "validation errors")

## Type Documentation

### Struct Documentation

```rust
/// Ordinary Least Squares (OLS) linear regression.
///
/// Fits a linear model by minimizing the residual sum of squares between
/// observed targets and predicted targets. The model equation is:
///
/// ```text
/// y = X β + ε
/// ```
///
/// where `β` is the coefficient vector and `ε` is random error.
///
/// # Solver
///
/// Uses normal equations: `β = (X^T X)^-1 X^T y` via Cholesky decomposition.
///
/// # Examples
///
/// ```
/// use aprender::prelude::*;
///
/// let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
/// let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);
///
/// let mut model = LinearRegression::new();
/// model.fit(&x, &y).unwrap();
/// let predictions = model.predict(&x);
/// ```
///
/// # Performance
///
/// - Time complexity: O(n²p + p³) where n = samples, p = features
/// - Space complexity: O(np)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearRegression {
    /// Coefficients for features (excluding intercept).
    coefficients: Option<Vector<f32>>,
    /// Intercept (bias) term.
    intercept: f32,
    /// Whether to fit an intercept.
    fit_intercept: bool,
}
```

**Location:** `src/linear_model/mod.rs:13-62`

**Elements:**
1. **Summary**: What the type represents
2. **Algorithm/Theory**: Mathematical foundation (for ML types)
3. **Examples**: How to create and use the type
4. **Performance**: Complexity, memory usage
5. **Field docs**: Document all fields (even private ones)

### Enum Documentation

```rust
/// Errors that can occur in aprender operations.
///
/// This enum represents all error conditions that can occur when using
/// aprender. Variants provide detailed context about what went wrong.
///
/// # Examples
///
/// ```
/// use aprender::error::AprenderError;
///
/// let err = AprenderError::DimensionMismatch {
///     expected: "100x10".to_string(),
///     actual: "50x10".to_string(),
/// };
///
/// println!("Error: {}", err);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub enum AprenderError {
    /// Matrix/vector dimensions don't match for the operation.
    DimensionMismatch {
        expected: String,
        actual: String,
    },

    /// Matrix is singular (non-invertible).
    SingularMatrix {
        det: f64,
    },

    /// Algorithm failed to converge within iteration limit.
    ConvergenceFailure {
        iterations: usize,
        final_loss: f64,
    },

    // ... more variants
}
```

**Location:** `src/error.rs:7-78`

**Elements:**
1. **Summary**: Purpose of the enum
2. **Examples**: Creating and using variants
3. **Variant docs**: Document each variant's meaning

### Trait Documentation

```rust
/// Primary trait for supervised learning estimators.
///
/// Estimators implement fit/predict/score following sklearn conventions.
/// Models that implement this trait can be used interchangeably in pipelines,
/// cross-validation, and hyperparameter tuning.
///
/// # Required Methods
///
/// - `fit()`: Train the model on labeled data
/// - `predict()`: Make predictions on new data
/// - `score()`: Evaluate model performance
///
/// # Examples
///
/// ```
/// use aprender::prelude::*;
///
/// fn train_and_evaluate<E: Estimator>(mut estimator: E) -> f32 {
///     let x = Matrix::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
///     let y = Vector::from_slice(&[3.0, 5.0, 7.0, 9.0]);
///
///     estimator.fit(&x, &y).unwrap();
///     estimator.score(&x, &y)
/// }
///
/// // Works with any Estimator
/// let model = LinearRegression::new();
/// let r2 = train_and_evaluate(model);
/// assert!(r2 > 0.99);
/// ```
pub trait Estimator {
    /// Fits the model to training data.
    fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()>;

    /// Predicts target values for input data.
    fn predict(&self, x: &Matrix<f32>) -> Vector<f32>;

    /// Computes the score (R² for regression, accuracy for classification).
    fn score(&self, x: &Matrix<f32>, y: &Vector<f32>) -> f32;
}
```

**Location:** `src/traits.rs:8-44`

**Elements:**
1. **Summary**: Purpose of the trait
2. **Context**: When to implement, design philosophy
3. **Required Methods**: List and explain each method
4. **Examples**: Generic function using the trait

## Doctests

Doctests are **executable examples** in documentation:

### Basic Doctest

```rust
/// Computes the dot product of two vectors.
///
/// # Examples
///
/// ```
/// use aprender::primitives::Vector;
///
/// let a = Vector::from_slice(&[1.0, 2.0, 3.0]);
/// let b = Vector::from_slice(&[4.0, 5.0, 6.0]);
///
/// let dot = a.dot(&b);
/// assert_eq!(dot, 32.0);  // 1*4 + 2*5 + 3*6 = 32
/// ```
pub fn dot(&self, other: &Vector<f32>) -> f32 {
    // Implementation...
}
```

**Run doctests:**
```bash
cargo test --doc              # Run all doctests
cargo test --doc -- linear    # Run doctests containing "linear"
```

### Doctest Attributes

```rust
/// Saves model to disk.
///
/// # Examples
///
/// ```no_run
/// # use aprender::prelude::*;
/// let model = LinearRegression::new();
/// model.save("model.bin").unwrap();  // Don't actually write file during test
/// ```
```

**Common attributes:**
- `no_run`: Compile but don't execute (for I/O operations)
- `ignore`: Skip this doctest entirely
- `should_panic`: Expect the code to panic

### Hidden Lines in Doctests

```rust
/// Computes R² score.
///
/// # Examples
///
/// ```
/// # use aprender::prelude::*;
/// # let x = Matrix::from_vec(2, 1, vec![1.0, 2.0]).unwrap();
/// # let y = Vector::from_slice(&[3.0, 5.0]);
/// # let mut model = LinearRegression::new();
/// # model.fit(&x, &y).unwrap();
/// let score = model.score(&x, &y);
/// assert!(score > 0.99);
/// ```
```

**Lines starting with `#` are hidden in rendered docs but executed in tests.**

Use for:
- Imports (`use aprender::prelude::*;`)
- Setup code (creating test data)
- Boilerplate that distracts from the example

## Documentation Patterns

### Pattern 1: Progressive Disclosure

Start simple, add complexity gradually:

```rust
/// K-Means clustering algorithm.
///
/// # Basic Example
///
/// ```
/// use aprender::prelude::*;
///
/// let data = Matrix::from_vec(4, 2, vec![
///     0.0, 0.0,
///     0.1, 0.1,
///     10.0, 10.0,
///     10.1, 10.1,
/// ]).unwrap();
///
/// let mut kmeans = KMeans::new(2);
/// kmeans.fit(&data).unwrap();
/// ```
///
/// # Advanced: Hyperparameter Tuning
///
/// ```
/// # use aprender::prelude::*;
/// # let data = Matrix::from_vec(4, 2, vec![0.0; 8]).unwrap();
/// let mut kmeans = KMeans::new(3)
///     .with_max_iter(500)
///     .with_tol(1e-6)
///     .with_random_state(42);
///
/// kmeans.fit(&data).unwrap();
/// let inertia = kmeans.inertia();
/// ```
```

### Pattern 2: Show Both Success and Failure

```rust
/// Loads a model from disk.
///
/// # Examples
///
/// ## Success
///
/// ```no_run
/// # use aprender::prelude::*;
/// let model = LinearRegression::load("model.bin").unwrap();
/// let predictions = model.predict(&x);
/// ```
///
/// ## Handling Errors
///
/// ```no_run
/// # use aprender::prelude::*;
/// match LinearRegression::load("model.bin") {
///     Ok(model) => println!("Loaded successfully"),
///     Err(e) => eprintln!("Failed to load: {}", e),
/// }
/// ```
```

### Pattern 3: Link to Related Items

```rust
/// Splits data into training and test sets.
///
/// See also:
/// - [`KFold`] for cross-validation splits
/// - [`cross_validate`] for complete cross-validation
///
/// [`KFold`]: crate::model_selection::KFold
/// [`cross_validate`]: crate::model_selection::cross_validate
```

Use intra-doc links to help users discover related functionality.

## Common Documentation Pitfalls

### Pitfall 1: Outdated Examples

```rust
// ❌ Example doesn't compile - API changed
/// # Examples
///
/// ```
/// let model = LinearRegression::new(true);  // Constructor signature changed!
/// model.train(&x, &y);  // Method renamed to fit()!
/// ```
```

**Prevention:** Run `cargo test --doc` regularly. Doctests prevent documentation rot.

### Pitfall 2: Missing Imports

```rust
// ❌ Example won't compile - missing imports
/// ```
/// let model = LinearRegression::new();  // Where does this come from?
/// ```
```

**Fix:**
```rust
// ✅ Show imports
/// ```
/// use aprender::prelude::*;
///
/// let model = LinearRegression::new();
/// ```
```

### Pitfall 3: Incomplete Examples

```rust
// ❌ Example doesn't show how to use the result
/// ```
/// let model = LinearRegression::new();
/// model.fit(&x, &y).unwrap();
/// // Now what?
/// ```
```

**Fix:**
```rust
// ✅ Complete workflow
/// ```
/// # use aprender::prelude::*;
/// # let x = Matrix::from_vec(2, 1, vec![1.0, 2.0]).unwrap();
/// # let y = Vector::from_slice(&[3.0, 5.0]);
/// let mut model = LinearRegression::new();
/// model.fit(&x, &y).unwrap();
///
/// // Make predictions
/// let predictions = model.predict(&x);
///
/// // Evaluate
/// let r2 = model.score(&x, &y);
/// println!("R² = {}", r2);
/// ```
```

### Pitfall 4: No Motivation

```rust
// ❌ Doesn't explain *why* you'd use this
/// Sets the tolerance parameter.
pub fn with_tolerance(mut self, tol: f32) -> Self { }
```

**Fix:**
```rust
// ✅ Explains purpose and impact
/// Sets the convergence tolerance.
///
/// Smaller values lead to more accurate solutions but require more iterations.
/// Larger values converge faster but may be less precise.
///
/// Default: 1e-4 (good for most use cases)
///
/// # Examples
///
/// ```
/// # use aprender::cluster::KMeans;
/// // High precision (slower)
/// let kmeans = KMeans::new(3).with_tol(1e-8);
///
/// // Fast convergence (less precise)
/// let kmeans = KMeans::new(3).with_tol(1e-2);
/// ```
pub fn with_tolerance(mut self, tol: f32) -> Self { }
```

### Pitfall 5: Assuming Knowledge

```rust
// ❌ Uses jargon without explanation
/// Uses k-means++ initialization with Lloyd's algorithm.
```

**Fix:**
```rust
// ✅ Explains concepts
/// Initializes centroids using k-means++ (smart initialization that spreads
/// centroids apart) then runs Lloyd's algorithm (iteratively assign points
/// to nearest centroid and recompute centroids).
```

## Documentation Checklist

Before merging code, verify:

- [ ] Module has `//!` documentation with example
- [ ] All public types have `///` documentation
- [ ] All public functions have:
  - [ ] Summary line
  - [ ] Example (that compiles and runs)
  - [ ] `# Errors` section (if returns `Result`)
  - [ ] `# Panics` section (if can panic)
  - [ ] `# Arguments` section (for complex parameters)
- [ ] Doctests compile and pass (`cargo test --doc`)
- [ ] Examples show complete workflow (imports, setup, usage)
- [ ] Links to related items (traits, types, functions)
- [ ] Performance notes (for algorithms and hot paths)

## Tools

### Generate Documentation

```bash
# Generate docs for your crate only (no dependencies)
cargo doc --no-deps --open

# Include private items (for internal docs)
cargo doc --document-private-items

# Check for broken links
cargo doc --no-deps 2>&1 | grep "warning: unresolved link"
```

### Test Documentation

```bash
# Run all doctests
cargo test --doc

# Run specific doctest
cargo test --doc -- linear_regression

# Show doctest output
cargo test --doc -- --nocapture
```

### Documentation Coverage

```bash
# Check which items lack documentation (requires nightly)
cargo +nightly rustdoc -- -Z unstable-options --show-coverage
```

## Summary

Good documentation is **code**—it must be maintained, tested, and refactored:

**Key principles:**
1. **Executable examples**: Use doctests to prevent documentation rot
2. **Progressive disclosure**: Start simple, add complexity
3. **Complete workflows**: Show imports, setup, and usage
4. **Explain why**: Motivation, trade-offs, when to use
5. **Consistent structure**: Follow standard sections (Args, Returns, Errors, Examples)
6. **Link related items**: Help users discover functionality
7. **Test regularly**: `cargo test --doc` catches broken examples

**Documentation sections (in order):**
1. Summary (one sentence)
2. Details (algorithm, approach)
3. Arguments
4. Returns
5. Errors
6. Panics
7. Examples
8. Performance

**Real-world examples:**
- `src/lib.rs:1-47` - Module-level documentation with Quick Start
- `src/linear_model/mod.rs:13-62` - Struct documentation with math and examples
- `src/traits.rs:8-44` - Trait documentation with generic examples
- `src/error.rs:7-78` - Enum documentation with variant descriptions

**Tools:**
- `cargo doc --no-deps --open` - Generate and view documentation
- `cargo test --doc` - Run doctests to verify examples
- `# hidden lines` - Hide boilerplate while keeping tests complete

Documentation is not an afterthought—it's an essential part of your API that ensures your code is **usable, maintainable, and discoverable**.

# Error Handling

Error handling is fundamental to building robust machine learning applications. Aprender uses Rust's type-safe error handling with rich context to help users quickly identify and resolve issues.

## Core Principles

### 1. Use Result<T> for Fallible Operations

**Rule**: Any operation that can fail returns `Result<T>` instead of panicking.

```rust
// ✅ GOOD: Returns Result for dimension check
pub fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
    if x.shape().0 != y.len() {
        return Err(AprenderError::DimensionMismatch {
            expected: format!("{}x? (samples match)", y.len()),
            actual: format!("{}x{}", x.shape().0, x.shape().1),
        });
    }
    // ... rest of implementation
    Ok(())
}

// ❌ BAD: Panics instead of returning error
pub fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) {
    assert_eq!(x.shape().0, y.len(), "Dimension mismatch!");  // Panic!
    // ...
}
```

**Why?** Users can handle errors gracefully instead of crashing their applications.

### 2. Provide Rich Error Context

**Rule**: Error messages should include enough context to debug the issue without looking at source code.

```rust
// ✅ GOOD: Detailed error with actual values
return Err(AprenderError::InvalidHyperparameter {
    param: "learning_rate".to_string(),
    value: format!("{}", lr),
    constraint: "must be > 0.0".to_string(),
});

// ❌ BAD: Vague error message
return Err("Invalid learning rate".into());
```

**Example output**:
```
Error: Invalid hyperparameter: learning_rate = -0.1, expected must be > 0.0
```

Users immediately understand:
- What parameter is wrong
- What value they provided
- What constraint was violated

### 3. Match Error Types to Failure Modes

**Rule**: Use specific error variants, not generic `Other`.

```rust
// ✅ GOOD: Specific error type
if x.shape().0 != y.len() {
    return Err(AprenderError::DimensionMismatch {
        expected: format!("samples={}", y.len()),
        actual: format!("samples={}", x.shape().0),
    });
}

// ❌ BAD: Generic error loses type information
if x.shape().0 != y.len() {
    return Err(AprenderError::Other("Shapes don't match".to_string()));
}
```

**Benefit**: Users can pattern match on specific errors for recovery strategies.

## AprenderError Design

### Error Variants

```rust
pub enum AprenderError {
    /// Matrix/vector dimensions incompatible for operation
    DimensionMismatch {
        expected: String,
        actual: String,
    },

    /// Matrix is singular (not invertible)
    SingularMatrix {
        det: f64,
    },

    /// Algorithm failed to converge
    ConvergenceFailure {
        iterations: usize,
        final_loss: f64,
    },

    /// Invalid hyperparameter value
    InvalidHyperparameter {
        param: String,
        value: String,
        constraint: String,
    },

    /// Compute backend unavailable
    BackendUnavailable {
        backend: String,
    },

    /// File I/O error
    Io(std::io::Error),

    /// Serialization error
    Serialization(String),

    /// Catch-all for other errors
    Other(String),
}
```

### When to Use Each Variant

| Variant | Use When | Example |
|---------|----------|---------|
| **DimensionMismatch** | Matrix/vector shapes incompatible | `fit(x: 100x5, y: len=50)` |
| **SingularMatrix** | Matrix cannot be inverted | Ridge regression with λ=0 on rank-deficient matrix |
| **ConvergenceFailure** | Iterative algorithm doesn't converge | Lasso with max_iter=10 insufficient |
| **InvalidHyperparameter** | Parameter violates constraint | `learning_rate = -0.1` (must be positive) |
| **BackendUnavailable** | Requested hardware unavailable | GPU operations on CPU-only machine |
| **Io** | File operations fail | Model file not found, permission denied |
| **Serialization** | Save/load fails | Corrupted model file |
| **Other** | Unexpected errors | Last resort, prefer specific variants |

### Rich Context Pattern

**Structure**: `{error_type}: {what} = {actual}, expected {constraint}`

```rust
// DimensionMismatch example
AprenderError::DimensionMismatch {
    expected: "100x10 (samples=100, features=10)",
    actual: "100x5 (samples=100, features=5)",
}
// Output: "Matrix dimension mismatch: expected 100x10 (samples=100, features=10), got 100x5 (samples=100, features=5)"

// InvalidHyperparameter example
AprenderError::InvalidHyperparameter {
    param: "n_clusters",
    value: "0",
    constraint: "must be >= 1",
}
// Output: "Invalid hyperparameter: n_clusters = 0, expected must be >= 1"
```

## Error Handling Patterns

### Pattern 1: Early Return with ?

**Use the ? operator** for error propagation:

```rust
pub fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
    // Validate dimensions
    self.validate_inputs(x, y)?;  // Early return if error

    // Check hyperparameters
    self.validate_hyperparameters()?;  // Early return if error

    // Perform training
    self.train_internal(x, y)?;  // Early return if error

    Ok(())
}

fn validate_inputs(&self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
    if x.shape().0 != y.len() {
        return Err(AprenderError::DimensionMismatch {
            expected: format!("samples={}", y.len()),
            actual: format!("samples={}", x.shape().0),
        });
    }
    Ok(())
}
```

**Benefits**:
- Clean, readable code
- Errors automatically propagate up the call stack
- Explicit Result types in signatures

### Pattern 2: Result Type Alias

**Use the crate-level Result alias**:

```rust
use crate::error::Result;  // = std::result::Result<T, AprenderError>

// ✅ GOOD: Concise type signature
pub fn predict(&self, x: &Matrix<f32>) -> Result<Vector<f32>> {
    // ...
}

// ❌ VERBOSE: Fully qualified type
pub fn predict(&self, x: &Matrix<f32>)
    -> std::result::Result<Vector<f32>, crate::error::AprenderError>
{
    // ...
}
```

### Pattern 3: Validate Early, Fail Fast

**Check preconditions at function entry**:

```rust
pub fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
    // 1. Validate inputs FIRST
    if x.shape().0 == 0 {
        return Err("Cannot fit on empty dataset".into());
    }

    if x.shape().0 != y.len() {
        return Err(AprenderError::DimensionMismatch {
            expected: format!("samples={}", y.len()),
            actual: format!("samples={}", x.shape().0),
        });
    }

    // 2. Validate hyperparameters
    if self.learning_rate <= 0.0 {
        return Err(AprenderError::InvalidHyperparameter {
            param: "learning_rate".to_string(),
            value: format!("{}", self.learning_rate),
            constraint: "> 0.0".to_string(),
        });
    }

    // 3. Proceed with training (all checks passed)
    self.train_internal(x, y)
}
```

**Benefits**:
- Errors caught before expensive computation
- Clear failure points
- Easy to test edge cases

### Pattern 4: Convert External Errors

**Use From trait** for automatic conversion:

```rust
impl From<std::io::Error> for AprenderError {
    fn from(err: std::io::Error) -> Self {
        AprenderError::Io(err)
    }
}

// Now you can use ? with io::Error
pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
    let file = File::create(path)?;  // io::Error → AprenderError automatically
    let writer = BufWriter::new(file);
    serde_json::to_writer(writer, self)?;  // Would need From for serde error
    Ok(())
}
```

### Pattern 5: Custom Error Messages with .map_err()

**Add context when converting errors**:

```rust
pub fn load_model(path: &str) -> Result<Model> {
    let file = File::open(path)
        .map_err(|e| AprenderError::Other(
            format!("Failed to open model file '{}': {}", path, e)
        ))?;

    let model: Model = serde_json::from_reader(file)
        .map_err(|e| AprenderError::Serialization(
            format!("Failed to deserialize model: {}", e)
        ))?;

    Ok(model)
}
```

## Real-World Examples from Aprender

### Example 1: Linear Regression Dimension Check

```rust
// From: src/linear_model/mod.rs
impl Estimator<f32, f32> for LinearRegression {
    fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
        let (n_samples, n_features) = x.shape();

        // Validate sample count matches
        if n_samples != y.len() {
            return Err(AprenderError::DimensionMismatch {
                expected: format!("{}x{}", y.len(), n_features),
                actual: format!("{}x{}", n_samples, n_features),
            });
        }

        // Validate non-empty
        if n_samples == 0 {
            return Err("Cannot fit on empty dataset".into());
        }

        // ... training logic
        Ok(())
    }
}
```

**Error message example**:
```
Error: Matrix dimension mismatch: expected 100x5, got 80x5
```

User immediately knows:
- Expected 100 samples, got 80
- Feature count (5) is correct
- Need to check training data creation

### Example 2: K-Means Hyperparameter Validation

```rust
// From: src/cluster/mod.rs
impl KMeans {
    pub fn new(n_clusters: usize) -> Result<Self> {
        if n_clusters == 0 {
            return Err(AprenderError::InvalidHyperparameter {
                param: "n_clusters".to_string(),
                value: "0".to_string(),
                constraint: "must be >= 1".to_string(),
            });
        }

        Ok(Self {
            n_clusters,
            max_iter: 300,
            tol: 1e-4,
            random_state: None,
            centroids: None,
        })
    }
}
```

**Usage**:
```rust
match KMeans::new(0) {
    Ok(_) => println!("Created K-Means"),
    Err(e) => println!("Error: {}", e),
    // Prints: "Error: Invalid hyperparameter: n_clusters = 0, expected must be >= 1"
}
```

### Example 3: Ridge Regression Singular Matrix

```rust
// From: src/linear_model/mod.rs
impl Estimator<f32, f32> for Ridge {
    fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
        // ... dimension checks ...

        // Compute X^T X + λI
        let xtx = x.transpose().matmul(&x);
        let regularized = xtx + self.alpha * Matrix::identity(n_features);

        // Attempt Cholesky decomposition (fails if singular)
        let cholesky = match regularized.cholesky() {
            Some(l) => l,
            None => {
                return Err(AprenderError::SingularMatrix {
                    det: 0.0,  // Approximate (actual computation expensive)
                });
            }
        };

        // ... solve system ...
        Ok(())
    }
}
```

**Error message**:
```
Error: Singular matrix detected: determinant = 0, cannot invert
```

**Recovery strategy**:
```rust
match ridge.fit(&x, &y) {
    Ok(()) => println!("Training succeeded"),
    Err(AprenderError::SingularMatrix { .. }) => {
        println!("Matrix is singular, try increasing regularization:");
        println!("  ridge.alpha = 1.0  (current: {})", ridge.alpha);
    }
    Err(e) => println!("Other error: {}", e),
}
```

### Example 4: Lasso Convergence Failure

```rust
// From: src/linear_model/mod.rs
impl Estimator<f32, f32> for Lasso {
    fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
        // ... setup ...

        for iter in 0..self.max_iter {
            let prev_coef = self.coefficients.clone();

            // Coordinate descent update
            self.update_coordinates(x, y);

            // Check convergence
            let change = compute_max_change(&self.coefficients, &prev_coef);
            if change < self.tol {
                return Ok(());  // Converged!
            }
        }

        // Did not converge
        Err(AprenderError::ConvergenceFailure {
            iterations: self.max_iter,
            final_loss: self.compute_loss(x, y),
        })
    }
}
```

**Error handling**:
```rust
match lasso.fit(&x, &y) {
    Ok(()) => println!("Training converged"),
    Err(AprenderError::ConvergenceFailure { iterations, final_loss }) => {
        println!("Warning: Did not converge after {} iterations", iterations);
        println!("Final loss: {:.4}", final_loss);
        println!("Try: lasso.max_iter = {}", iterations * 2);
    }
    Err(e) => println!("Error: {}", e),
}
```

## User-Facing Error Handling

### Pattern: Match on Error Types

```rust
use aprender::classification::KNearestNeighbors;
use aprender::error::AprenderError;

fn train_model(x: &Matrix<f32>, y: &Vec<i32>) {
    let mut knn = KNearestNeighbors::new(5);

    match knn.fit(x, y) {
        Ok(()) => println!("✅ Training succeeded"),

        Err(AprenderError::DimensionMismatch { expected, actual }) => {
            eprintln!("❌ Dimension mismatch:");
            eprintln!("   Expected: {}", expected);
            eprintln!("   Got:      {}", actual);
            eprintln!("   Fix: Check your training data shapes");
        }

        Err(AprenderError::InvalidHyperparameter { param, value, constraint }) => {
            eprintln!("❌ Invalid parameter: {} = {}", param, value);
            eprintln!("   Constraint: {}", constraint);
            eprintln!("   Fix: Adjust hyperparameter value");
        }

        Err(e) => {
            eprintln!("❌ Unexpected error: {}", e);
        }
    }
}
```

### Pattern: Propagate with Context

```rust
fn load_and_train(model_path: &str, data_path: &str) -> Result<Model> {
    // Load pre-trained model
    let mut model = Model::load(model_path)
        .map_err(|e| format!("Failed to load model from '{}': {}", model_path, e))?;

    // Load training data
    let (x, y) = load_data(data_path)
        .map_err(|e| format!("Failed to load data from '{}': {}", data_path, e))?;

    // Fine-tune model
    model.fit(&x, &y)
        .map_err(|e| format!("Training failed: {}", e))?;

    Ok(model)
}
```

### Pattern: Recover from Specific Errors

```rust
fn robust_training(x: &Matrix<f32>, y: &Vector<f32>) -> Result<Ridge> {
    let mut ridge = Ridge::new(0.1);  // Small regularization

    match ridge.fit(x, y) {
        Ok(()) => return Ok(ridge),

        // Recovery: Increase regularization if matrix is singular
        Err(AprenderError::SingularMatrix { .. }) => {
            println!("Warning: Matrix singular with α=0.1, trying α=1.0");
            ridge.alpha = 1.0;
            ridge.fit(x, y)?;  // Retry with stronger regularization
            Ok(ridge)
        }

        // Propagate other errors
        Err(e) => Err(e),
    }
}
```

## Testing Error Conditions

### Test Each Error Variant

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dimension_mismatch_error() {
        let x = Matrix::from_vec(100, 5, vec![0.0; 500]).unwrap();
        let y = Vector::from_vec(vec![0.0; 80]);  // Wrong size!

        let mut lr = LinearRegression::new();
        let result = lr.fit(&x, &y);

        assert!(result.is_err());
        match result.unwrap_err() {
            AprenderError::DimensionMismatch { expected, actual } => {
                assert!(expected.contains("80"));
                assert!(actual.contains("100"));
            }
            _ => panic!("Expected DimensionMismatch error"),
        }
    }

    #[test]
    fn test_invalid_hyperparameter_error() {
        let result = KMeans::new(0);  // Invalid: n_clusters must be >= 1

        assert!(result.is_err());
        match result.unwrap_err() {
            AprenderError::InvalidHyperparameter { param, value, constraint } => {
                assert_eq!(param, "n_clusters");
                assert_eq!(value, "0");
                assert!(constraint.contains(">= 1"));
            }
            _ => panic!("Expected InvalidHyperparameter error"),
        }
    }

    #[test]
    fn test_convergence_failure_error() {
        let x = Matrix::from_vec(10, 5, vec![1.0; 50]).unwrap();
        let y = Vector::from_vec(vec![1.0; 10]);

        let mut lasso = Lasso::new(0.1)
            .with_max_iter(1);  // Force non-convergence

        let result = lasso.fit(&x, &y);

        assert!(result.is_err());
        match result.unwrap_err() {
            AprenderError::ConvergenceFailure { iterations, .. } => {
                assert_eq!(iterations, 1);
            }
            _ => panic!("Expected ConvergenceFailure error"),
        }
    }
}
```

## Common Pitfalls

### Pitfall 1: Using panic!() Instead of Result

```rust
// ❌ BAD: Crashes user's application
pub fn predict(&self, x: &Matrix<f32>) -> Vector<f32> {
    assert!(self.is_fitted(), "Model not fitted!");  // Panic!
    // ...
}

// ✅ GOOD: Returns error user can handle
pub fn predict(&self, x: &Matrix<f32>) -> Result<Vector<f32>> {
    if !self.is_fitted() {
        return Err("Model not fitted, call fit() first".into());
    }
    // ...
    Ok(predictions)
}
```

### Pitfall 2: Swallowing Errors

```rust
// ❌ BAD: Error information lost
pub fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
    if let Err(_) = self.validate_inputs(x, y) {
        return Err("Validation failed".into());  // Context lost!
    }
    // ...
}

// ✅ GOOD: Propagate full error
pub fn fit(&mut self, x: &Matrix<f32>, y: &Vector<f32>) -> Result<()> {
    self.validate_inputs(x, y)?;  // Full error propagated
    // ...
}
```

### Pitfall 3: Generic Other Errors

```rust
// ❌ BAD: Loses type information
if n_clusters == 0 {
    return Err(AprenderError::Other("n_clusters must be >= 1".into()));
}

// ✅ GOOD: Specific error variant
if n_clusters == 0 {
    return Err(AprenderError::InvalidHyperparameter {
        param: "n_clusters".to_string(),
        value: "0".to_string(),
        constraint: ">= 1".to_string(),
    });
}
```

### Pitfall 4: Unclear Error Messages

```rust
// ❌ BAD: Not actionable
return Err("Invalid input".into());

// ✅ GOOD: Specific and actionable
return Err(AprenderError::DimensionMismatch {
    expected: format!("samples={}, features={}", expected_samples, expected_features),
    actual: format!("samples={}, features={}", x.shape().0, x.shape().1),
});
```

## Best Practices Summary

| Practice | Do | Don't |
|----------|-----|-------|
| **Return types** | Use `Result<T>` for fallible operations | Use `panic!()` or `unwrap()` in library code |
| **Error variants** | Use specific error types | Use generic `Other` variant |
| **Error messages** | Include actual values and context | Use vague messages like "Invalid input" |
| **Propagation** | Use `?` operator | Manually match and re-wrap errors |
| **Validation** | Check preconditions early | Validate late, fail deep in call stack |
| **Testing** | Test each error variant | Only test happy path |
| **Recovery** | Match on specific error types | Ignore error details |

## Further Reading

- **Rust Book**: [Error Handling Chapter](https://doc.rust-lang.org/book/ch09-00-error-handling.html)
- **Rust By Example**: [Error Handling](https://doc.rust-lang.org/rust-by-example/error.html)
- **Rust API Guidelines**: [Error Design](https://rust-lang.github.io/api-guidelines/interoperability.html#error-types-are-meaningful-and-well-behaved-c-good-err)

## Related Chapters

- [API Design](./api-design.md) - How Result fits into API design
- [Type Safety](./type-safety.md) - Using types to prevent errors
- [Testing](../methodology/test-first-philosophy.md) - Testing error paths

## Summary

| Concept | Key Takeaway |
|---------|--------------|
| **Result<T>** | All fallible operations return Result, never panic |
| **Rich context** | Errors include actual values, expected values, constraints |
| **Specific variants** | Use DimensionMismatch, InvalidHyperparameter, not generic Other |
| **Early validation** | Check preconditions at function entry, fail fast |
| **? operator** | Use for clean error propagation |
| **Pattern matching** | Users match on error types for recovery strategies |
| **Testing** | Test each error variant with targeted tests |

Excellent error handling makes the difference between a frustrating library and a delightful one. Users should always know what went wrong and how to fix it.

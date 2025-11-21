# Builder Pattern

The **Builder Pattern** is a creational design pattern that constructs complex objects with many optional parameters. In Rust ML libraries, it's the standard way to create estimators with sensible defaults while allowing customization.

## Why Use the Builder Pattern?

Machine learning models have many hyperparameters, most of which have good defaults:

```rust
// Without builder: telescoping constructor hell
let model = KMeans::new(
    3,           // n_clusters (required)
    300,         // max_iter
    1e-4,        // tol
    Some(42),    // random_state
);
// Which parameter was which? Hard to remember!

// With builder: clear, self-documenting, extensible
let model = KMeans::new(3)
    .with_max_iter(300)
    .with_tol(1e-4)
    .with_random_state(42);
// Clear intent, sensible defaults for omitted parameters
```

**Benefits:**
1. **Sensible defaults**: Only specify what differs from defaults
2. **Self-documenting**: Method names make intent clear
3. **Extensible**: Add new parameters without breaking existing code
4. **Type-safe**: Compile-time verification of parameter types
5. **Chainable**: Fluent API for configuring complex objects

## Implementation Pattern

### Basic Structure

```rust
pub struct KMeans {
    // Required parameter
    n_clusters: usize,

    // Optional parameters with defaults
    max_iter: usize,
    tol: f32,
    random_state: Option<u64>,

    // State (None until fitted)
    centroids: Option<Matrix<f32>>,
}

impl KMeans {
    /// Creates a new K-Means with required parameters and sensible defaults.
    #[must_use]  // ← CRITICAL: Warn if result is unused
    pub fn new(n_clusters: usize) -> Self {
        Self {
            n_clusters,
            max_iter: 300,          // Default from sklearn
            tol: 1e-4,              // Default from sklearn
            random_state: None,     // Default: non-deterministic
            centroids: None,        // Not fitted yet
        }
    }

    /// Sets the maximum number of iterations.
    #[must_use]  // ← Consuming self, must use return value
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self  // Return self for chaining
    }

    /// Sets the convergence tolerance.
    #[must_use]
    pub fn with_tol(mut self, tol: f32) -> Self {
        self.tol = tol;
        self
    }

    /// Sets the random seed for reproducibility.
    #[must_use]
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}
```

**Key elements:**
- `new()` takes only required parameters
- `with_*()` methods set optional parameters
- Methods consume `self` and return `Self` for chaining
- `#[must_use]` attribute warns if result is discarded

### Usage

```rust
// Use defaults
let mut kmeans = KMeans::new(3);
kmeans.fit(&data)?;

// Customize hyperparameters
let mut kmeans = KMeans::new(3)
    .with_max_iter(500)
    .with_tol(1e-5)
    .with_random_state(42);
kmeans.fit(&data)?;

// Can store builder and modify later
let builder = KMeans::new(3)
    .with_max_iter(500);
// Later...
let mut model = builder.with_random_state(42);
model.fit(&data)?;
```

## Real-World Examples from aprender

### Example 1: LogisticRegression

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticRegression {
    coefficients: Option<Vector<f32>>,
    intercept: f32,
    learning_rate: f32,
    max_iter: usize,
    tol: f32,
}

impl LogisticRegression {
    pub fn new() -> Self {
        Self {
            coefficients: None,
            intercept: 0.0,
            learning_rate: 0.01,    // Default
            max_iter: 1000,         // Default
            tol: 1e-4,              // Default
        }
    }

    pub fn with_learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn with_tolerance(mut self, tol: f32) -> Self {
        self.tol = tol;
        self
    }
}

// Usage
let mut model = LogisticRegression::new()
    .with_learning_rate(0.1)
    .with_max_iter(2000)
    .with_tolerance(1e-6);
model.fit(&x, &y)?;
```

**Location:** `src/classification/mod.rs:60-96`

### Example 2: DecisionTreeRegressor with Validation

```rust
impl DecisionTreeRegressor {
    pub fn new() -> Self {
        Self {
            tree: None,
            max_depth: None,         // None = unlimited
            min_samples_split: 2,    // Minimum valid value
            min_samples_leaf: 1,     // Minimum valid value
        }
    }

    pub fn with_max_depth(mut self, depth: usize) -> Self {
        self.max_depth = Some(depth);
        self
    }

    /// Sets minimum samples to split (enforces minimum of 2).
    pub fn with_min_samples_split(mut self, min_samples: usize) -> Self {
        self.min_samples_split = min_samples.max(2);  // ← Validation!
        self
    }

    /// Sets minimum samples per leaf (enforces minimum of 1).
    pub fn with_min_samples_leaf(mut self, min_samples: usize) -> Self {
        self.min_samples_leaf = min_samples.max(1);  // ← Validation!
        self
    }
}

// Usage - invalid values are coerced to valid ranges
let tree = DecisionTreeRegressor::new()
    .with_min_samples_split(0);  // Will be coerced to 2
```

**Key insight:** Builder methods can validate and coerce parameters to valid ranges.

**Location:** `src/tree/mod.rs:153-192`

### Example 3: StandardScaler with Boolean Flags

```rust
impl StandardScaler {
    #[must_use]
    pub fn new() -> Self {
        Self {
            mean: None,
            std: None,
            with_mean: true,   // Default: center data
            with_std: true,    // Default: scale data
        }
    }

    #[must_use]
    pub fn with_mean(mut self, with_mean: bool) -> Self {
        self.with_mean = with_mean;
        self
    }

    #[must_use]
    pub fn with_std(mut self, with_std: bool) -> Self {
        self.with_std = with_std;
        self
    }
}

// Usage: disable centering but keep scaling
let mut scaler = StandardScaler::new()
    .with_mean(false)
    .with_std(true);
scaler.fit_transform(&data)?;
```

**Location:** `src/preprocessing/mod.rs:84-111`

### Example 4: LinearRegression - Minimal Builder

```rust
impl LinearRegression {
    #[must_use]
    pub fn new() -> Self {
        Self {
            coefficients: None,
            intercept: 0.0,
            fit_intercept: true,  // Default: fit intercept
        }
    }

    #[must_use]
    pub fn with_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

// Usage
let mut model = LinearRegression::new();              // Use defaults
let mut model = LinearRegression::new()
    .with_intercept(false);                           // No intercept
```

**Key insight:** Even models with few parameters benefit from builder pattern for clarity and extensibility.

**Location:** `src/linear_model/mod.rs:70-86`

## The `#[must_use]` Attribute

The `#[must_use]` attribute is **CRITICAL** for builder methods:

```rust
#[must_use]
pub fn with_max_iter(mut self, max_iter: usize) -> Self {
    self.max_iter = max_iter;
    self
}
```

### Why `#[must_use]` Matters

Without it, this bug compiles silently:

```rust
// BUG: Result of with_max_iter() is discarded!
let mut model = KMeans::new(3);
model.with_max_iter(500);  // ← Does NOTHING! Returns modified copy
model.fit(&data)?;         // ← Uses default max_iter=300, not 500

// Correct usage (compiler warns without #[must_use])
let mut model = KMeans::new(3)
    .with_max_iter(500);   // ← Assigns modified copy
model.fit(&data)?;         // ← Uses max_iter=500
```

**Always use `#[must_use]` on:**
1. `new()` constructors (warn if unused)
2. All `with_*()` builder methods (consuming self)
3. Methods that return `Self` without side effects

### Anti-Pattern in Codebase

`src/classification/mod.rs:80-96` is **missing `#[must_use]`**:

```rust
// ❌ MISSING #[must_use] - should be fixed
pub fn with_learning_rate(mut self, lr: f32) -> Self {
    self.learning_rate = lr;
    self
}
```

This allows the silent bug above to compile without warnings.

## When to Use vs. Not Use

### Use Builder Pattern When:

1. **Many optional parameters** (3+ optional parameters)
   ```rust
   KMeans::new(3)
       .with_max_iter(300)
       .with_tol(1e-4)
       .with_random_state(42)
   ```

2. **Sensible defaults exist** (sklearn conventions)
   ```rust
   // Most users don't need to change max_iter
   KMeans::new(3)  // Uses max_iter=300 by default
   ```

3. **Future extensibility** (easy to add parameters without breaking API)
   ```rust
   // Later: add with_n_init() without breaking existing code
   KMeans::new(3)
       .with_max_iter(300)
       .with_n_init(10)  // New parameter
   ```

### Don't Use Builder Pattern When:

1. **All parameters are required** (use regular constructor)
   ```rust
   // ✅ Simple constructor - no builder needed
   Matrix::from_vec(rows, cols, data)
   ```

2. **Only one or two parameters** (constructor is clear enough)
   ```rust
   // ✅ No builder needed
   Vector::from_vec(data)
   ```

3. **Configuration is complex** (use dedicated config struct)
   ```rust
   // For very complex configuration (10+ parameters)
   struct KMeansConfig { /* ... */ }
   KMeans::from_config(config)
   ```

## Common Pitfalls

### Pitfall 1: Mutable Reference Instead of Consuming Self

```rust
// ❌ WRONG: Takes &mut self, breaks chaining
pub fn with_max_iter(&mut self, max_iter: usize) {
    self.max_iter = max_iter;
}

// Can't chain!
let mut model = KMeans::new(3);
model.with_max_iter(500);            // No return value
model.with_tol(1e-4);                // Separate call
model.with_random_state(42);         // Can't chain

// ✅ CORRECT: Consumes self, returns Self
pub fn with_max_iter(mut self, max_iter: usize) -> Self {
    self.max_iter = max_iter;
    self
}

// Can chain!
let mut model = KMeans::new(3)
    .with_max_iter(500)
    .with_tol(1e-4)
    .with_random_state(42);
```

### Pitfall 2: Forgetting to Assign Result

```rust
// ❌ BUG: Creates builder but doesn't assign
KMeans::new(3)
    .with_max_iter(500);  // ← Result dropped!

let mut model = ???;  // Where's the model?

// ✅ CORRECT: Assign to variable
let mut model = KMeans::new(3)
    .with_max_iter(500);
```

### Pitfall 3: Modifying After Construction

```rust
// ❌ WRONG: Trying to modify after construction
let mut model = KMeans::new(3);
model.with_max_iter(500);  // ← Returns new instance, doesn't modify in place

// ✅ CORRECT: Rebuild with new parameters
let model = KMeans::new(3);
let model = model.with_max_iter(500);  // Reassign

// Or chain at construction:
let mut model = KMeans::new(3)
    .with_max_iter(500);
```

### Pitfall 4: Mixing Mutable and Immutable

```rust
// ❌ INCONSISTENT: Don't do this
pub fn new() -> Self { /* ... */ }
pub fn with_max_iter(&mut self, max_iter: usize) { /* ... */ }  // Mutable ref
pub fn with_tol(mut self, tol: f32) -> Self { /* ... */ }       // Consuming

// ✅ CONSISTENT: All builders consume self
pub fn new() -> Self { /* ... */ }
pub fn with_max_iter(mut self, max_iter: usize) -> Self { /* ... */ }
pub fn with_tol(mut self, tol: f32) -> Self { /* ... */ }
```

## Pattern Comparison

### Telescoping Constructors

```rust
// ❌ Telescoping constructors - hard to read, not extensible
impl KMeans {
    pub fn new(n_clusters: usize) -> Self { /* ... */ }
    pub fn new_with_iter(n_clusters: usize, max_iter: usize) -> Self { /* ... */ }
    pub fn new_with_iter_tol(n_clusters: usize, max_iter: usize, tol: f32) -> Self { /* ... */ }
    pub fn new_with_all(n_clusters: usize, max_iter: usize, tol: f32, seed: u64) -> Self { /* ... */ }
}

// Which constructor do I use?
let model = KMeans::new_with_iter_tol(3, 500, 1e-5);  // But I also want random_state!
```

### Setter Methods (Java-style)

```rust
// ❌ Mutable setters - verbose, can't validate state until fit()
impl KMeans {
    pub fn new(n_clusters: usize) -> Self { /* ... */ }
    pub fn set_max_iter(&mut self, max_iter: usize) { /* ... */ }
    pub fn set_tol(&mut self, tol: f32) { /* ... */ }
}

// Verbose, no chaining
let mut model = KMeans::new(3);
model.set_max_iter(500);
model.set_tol(1e-5);
model.set_random_state(42);
```

### Builder Pattern (Rust Idiom)

```rust
// ✅ Builder pattern - clear, chainable, extensible
impl KMeans {
    pub fn new(n_clusters: usize) -> Self { /* ... */ }
    pub fn with_max_iter(mut self, max_iter: usize) -> Self { /* ... */ }
    pub fn with_tol(mut self, tol: f32) -> Self { /* ... */ }
    pub fn with_random_state(mut self, seed: u64) -> Self { /* ... */ }
}

// Clear, chainable, self-documenting
let mut model = KMeans::new(3)
    .with_max_iter(500)
    .with_tol(1e-5)
    .with_random_state(42);
```

## Advanced: Typestate Pattern

For **compile-time guarantees** of correct usage, combine builder with typestate:

```rust
// Track whether model is fitted at compile time
pub struct Unfitted;
pub struct Fitted;

pub struct KMeans<State = Unfitted> {
    n_clusters: usize,
    centroids: Option<Matrix<f32>>,
    _state: PhantomData<State>,
}

impl KMeans<Unfitted> {
    pub fn new(n_clusters: usize) -> Self { /* ... */ }

    pub fn fit(self, data: &Matrix<f32>) -> Result<KMeans<Fitted>> {
        // Consumes unfitted model, returns fitted model
    }
}

impl KMeans<Fitted> {
    pub fn predict(&self, data: &Matrix<f32>) -> Vec<usize> {
        // Only available on fitted models
    }
}

// Usage
let model = KMeans::new(3);
// model.predict(&data);  // ← Compile error! Not fitted
let model = model.fit(&train_data)?;
let predictions = model.predict(&test_data);  // ✅ Compiles
```

**Trade-off:** More type safety but more complex API. Use only when compile-time guarantees are critical.

## Integration with Default Trait

Provide `Default` implementation when all parameters are optional:

```rust
impl Default for KMeans {
    fn default() -> Self {
        Self::new(8)  // sklearn default for n_clusters
    }
}

// Usage
let mut model = KMeans::default()
    .with_max_iter(500);
```

**When to implement `Default`:**
- All parameters have reasonable defaults (including "required" ones)
- Default values match sklearn conventions
- Useful for generic code that needs `T: Default`

**When NOT to implement `Default`:**
- Some parameters don't have sensible defaults (e.g., `n_clusters` is somewhat arbitrary)
- Could mislead users about what values to use

## Testing Builder Methods

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_defaults() {
        let model = KMeans::new(3);
        assert_eq!(model.n_clusters, 3);
        assert_eq!(model.max_iter, 300);
        assert_eq!(model.tol, 1e-4);
        assert_eq!(model.random_state, None);
    }

    #[test]
    fn test_builder_chaining() {
        let model = KMeans::new(3)
            .with_max_iter(500)
            .with_tol(1e-5)
            .with_random_state(42);

        assert_eq!(model.max_iter, 500);
        assert_eq!(model.tol, 1e-5);
        assert_eq!(model.random_state, Some(42));
    }

    #[test]
    fn test_builder_validation() {
        let tree = DecisionTreeRegressor::new()
            .with_min_samples_split(0);  // Invalid, should be coerced

        assert_eq!(tree.min_samples_split, 2);  // Coerced to minimum
    }
}
```

## Summary

The Builder Pattern is the **standard idiom** for configuring ML models in Rust:

**Key principles:**
1. `new()` takes only required parameters with sensible defaults
2. `with_*()` methods consume `self` and return `Self` for chaining
3. Always use `#[must_use]` attribute on builders
4. Validate parameters in builders when possible
5. Follow sklearn defaults for ML hyperparameters
6. Implement `Default` when all parameters are optional

**Why it works:**
- Rust's ownership system makes consuming builders efficient (no copies)
- Method chaining creates clear, self-documenting configuration
- Easy to extend without breaking existing code
- Type system enforces correct usage

**Real-world examples:**
- `src/cluster/mod.rs:77-112` - KMeans with multiple hyperparameters
- `src/linear_model/mod.rs:70-86` - LinearRegression with minimal builder
- `src/tree/mod.rs:153-192` - DecisionTreeRegressor with validation
- `src/preprocessing/mod.rs:84-111` - StandardScaler with boolean flags

The builder pattern is essential for creating ergonomic, maintainable ML APIs in Rust.

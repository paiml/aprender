# The RED-GREEN-REFACTOR Cycle

The **RED-GREEN-REFACTOR** cycle is the heartbeat of EXTREME TDD. Every feature, every function, every line of production code follows this exact three-phase cycle.

## The Three Phases

```
┌─────────────┐
│     RED     │  Write failing tests first
└──────┬──────┘
       │
       ↓
┌─────────────┐
│    GREEN    │  Implement minimally to pass tests
└──────┬──────┘
       │
       ↓
┌─────────────┐
│  REFACTOR   │  Improve quality with test safety net
└──────┬──────┘
       │
       ↓ (repeat for next feature)
```

## Phase 1: RED - Write Failing Tests

**Goal:** Create tests that define the desired behavior BEFORE writing implementation.

### Rules
1. ✅ Write tests BEFORE any implementation code
2. ✅ Run tests and verify they FAIL (for the right reason)
3. ✅ Tests should fail because feature doesn't exist, not because of syntax errors
4. ✅ Write multiple tests covering different scenarios

### Real Example: Cross-Validation Implementation

**CYCLE 1: train_test_split - RED Phase**

First, we created the failing tests in `src/model_selection/mod.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::{Matrix, Vector};

    #[test]
    fn test_train_test_split_basic() {
        let x = Matrix::from_vec(10, 2, vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0,
        ]).unwrap();
        let y = Vector::from_vec(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);

        let (x_train, x_test, y_train, y_test) =
            train_test_split(&x, &y, 0.2, None).expect("Split failed");

        // 80/20 split
        assert_eq!(x_train.shape().0, 8);
        assert_eq!(x_test.shape().0, 2);
        assert_eq!(y_train.len(), 8);
        assert_eq!(y_test.len(), 2);
    }

    #[test]
    fn test_train_test_split_reproducible() {
        let x = Matrix::from_vec(10, 2, vec![/* ... */]).unwrap();
        let y = Vector::from_vec(vec![/* ... */]);

        // Same seed = same split
        let (_, _, y_train1, _) = train_test_split(&x, &y, 0.3, Some(42)).unwrap();
        let (_, _, y_train2, _) = train_test_split(&x, &y, 0.3, Some(42)).unwrap();

        assert_eq!(y_train1.as_slice(), y_train2.as_slice());
    }

    #[test]
    fn test_train_test_split_different_seeds() {
        let x = Matrix::from_vec(100, 2, vec![/* ... */]).unwrap();
        let y = Vector::from_vec(vec![/* ... */]);

        // Different seeds = different splits
        let (_, _, y_train1, _) = train_test_split(&x, &y, 0.3, Some(42)).unwrap();
        let (_, _, y_train2, _) = train_test_split(&x, &y, 0.3, Some(123)).unwrap();

        assert_ne!(y_train1.as_slice(), y_train2.as_slice());
    }

    #[test]
    fn test_train_test_split_invalid_test_size() {
        let x = Matrix::from_vec(10, 2, vec![/* ... */]).unwrap();
        let y = Vector::from_vec(vec![/* ... */]);

        // test_size must be between 0 and 1
        assert!(train_test_split(&x, &y, 1.5, None).is_err());
        assert!(train_test_split(&x, &y, -0.1, None).is_err());
    }
}
```

**Verification (RED Phase):**
```bash
$ cargo test train_test_split
   Compiling aprender v0.1.0
error[E0425]: cannot find function `train_test_split` in this scope
  --> src/model_selection/mod.rs:12:9

# PERFECT! Tests fail because function doesn't exist yet ✅
```

**Result:** 4 failing tests (expected - feature not implemented)

### Key Principle: Fail for the Right Reason

```rust
// ❌ BAD: Test fails due to typo
#[test]
fn test_example() {
    let result = train_tset_split();  // Typo!
    assert_eq!(result, expected);
}

// ✅ GOOD: Test fails because feature doesn't exist
#[test]
fn test_example() {
    let result = train_test_split(&x, &y, 0.2, None);  // Compiles, but fails
    assert_eq!(result, expected);  // Assertion fails - function not implemented
}
```

## Phase 2: GREEN - Minimal Implementation

**Goal:** Write JUST enough code to make tests pass. No more, no less.

### Rules
1. ✅ Implement the simplest solution that makes tests pass
2. ✅ Avoid premature optimization
3. ✅ Don't add "future-proofing" features
4. ✅ Run tests after each change
5. ✅ Stop when all tests pass

### Real Example: train_test_split - GREEN Phase

We implemented the minimal solution:

```rust
#[allow(clippy::type_complexity)]
pub fn train_test_split(
    x: &Matrix<f32>,
    y: &Vector<f32>,
    test_size: f32,
    random_state: Option<u64>,
) -> Result<(Matrix<f32>, Matrix<f32>, Vector<f32>, Vector<f32>), String> {
    // Validation
    if test_size <= 0.0 || test_size >= 1.0 {
        return Err("test_size must be between 0 and 1".to_string());
    }

    let n_samples = x.shape().0;
    let n_test = (n_samples as f32 * test_size).round() as usize;
    let n_train = n_samples - n_test;

    // Create shuffled indices
    let mut indices: Vec<usize> = (0..n_samples).collect();

    // Shuffle if needed
    if let Some(seed) = random_state {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        use rand::seq::SliceRandom;
        indices.shuffle(&mut rng);
    } else {
        use rand::seq::SliceRandom;
        indices.shuffle(&mut rand::thread_rng());
    }

    // Split indices
    let train_idx = &indices[..n_train];
    let test_idx = &indices[n_train..];

    // Extract data
    let (x_train, y_train) = extract_samples(x, y, train_idx);
    let (x_test, y_test) = extract_samples(x, y, test_idx);

    Ok((x_train, x_test, y_train, y_test))
}
```

**Verification (GREEN Phase):**
```bash
$ cargo test train_test_split
   Compiling aprender v0.1.0
    Finished test [unoptimized + debuginfo] target(s) in 2.34s
     Running unittests src/lib.rs

running 4 tests
test model_selection::tests::test_train_test_split_basic ... ok
test model_selection::tests::test_train_test_split_reproducible ... ok
test model_selection::tests::test_train_test_split_different_seeds ... ok
test model_selection::tests::test_train_test_split_invalid_test_size ... ok

test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured

# SUCCESS! All tests pass ✅
```

**Result:** Tests: 169 total (165 + 4 new) ✅

### Avoiding Over-Engineering

```rust
// ❌ OVER-ENGINEERED: Adding features not required by tests
pub fn train_test_split(
    x: &Matrix<f32>,
    y: &Vector<f32>,
    test_size: f32,
    random_state: Option<u64>,
    stratify: bool,  // ❌ Not tested!
    shuffle_method: ShuffleMethod,  // ❌ Not needed!
    cache_results: bool,  // ❌ Premature optimization!
) -> Result<Split, Error> {
    // Complex caching logic...
    // Multiple shuffle algorithms...
    // Stratification logic...
}

// ✅ MINIMAL: Just what tests require
pub fn train_test_split(
    x: &Matrix<f32>,
    y: &Vector<f32>,
    test_size: f32,
    random_state: Option<u64>,
) -> Result<(Matrix<f32>, Matrix<f32>, Vector<f32>, Vector<f32>), String> {
    // Simple, clear implementation
}
```

## Phase 3: REFACTOR - Improve with Confidence

**Goal:** Improve code quality while maintaining all passing tests.

### Rules
1. ✅ All tests must continue passing
2. ✅ Add unit tests for edge cases
3. ✅ Run clippy and fix ALL warnings
4. ✅ Check cyclomatic complexity (≤10 target)
5. ✅ Add documentation
6. ✅ Run mutation tests
7. ✅ Optimize if needed (profile first)

### Real Example: train_test_split - REFACTOR Phase

**Step 1: Run Clippy**

```bash
$ cargo clippy -- -D warnings
warning: very complex type used. Consider factoring parts into `type` definitions
  --> src/model_selection/mod.rs:148:6
   |
   | pub fn train_test_split(
   |        ^^^^^^^^^^^^^^^^
```

**Fix:** Add allow annotation for idiomatic Rust tuple return:

```rust
#[allow(clippy::type_complexity)]
pub fn train_test_split(/* ... */) -> Result<(Matrix<f32>, Matrix<f32>, Vector<f32>, Vector<f32>), String> {
    // ...
}
```

**Step 2: Run Format Check**

```bash
$ cargo fmt --check
Diff in /home/noah/src/aprender/src/model_selection/mod.rs

$ cargo fmt
# Auto-format all code
```

**Step 3: Check Complexity**

```bash
$ pmat analyze complexity src/model_selection/
Function: train_test_split - Complexity: 4 ✅
Function: extract_samples - Complexity: 3 ✅

All functions ≤10 ✅
```

**Step 4: Add Documentation**

```rust
/// Splits data into random train and test subsets.
///
/// # Arguments
///
/// * `x` - Feature matrix of shape (n_samples, n_features)
/// * `y` - Target vector of length n_samples
/// * `test_size` - Proportion of dataset to include in test split (0.0 to 1.0)
/// * `random_state` - Seed for reproducible random splits
///
/// # Returns
///
/// Tuple of (x_train, x_test, y_train, y_test)
///
/// # Examples
///
/// ```
/// use aprender::model_selection::train_test_split;
/// use aprender::primitives::{Matrix, Vector};
///
/// let x = Matrix::from_vec(10, 2, vec![/* ... */]).unwrap();
/// let y = Vector::from_vec(vec![/* ... */]);
///
/// let (x_train, x_test, y_train, y_test) =
///     train_test_split(&x, &y, 0.2, Some(42)).unwrap();
///
/// assert_eq!(x_train.shape().0, 8);  // 80% train
/// assert_eq!(x_test.shape().0, 2);   // 20% test
/// ```
#[allow(clippy::type_complexity)]
pub fn train_test_split(/* ... */) {
    // ...
}
```

**Step 5: Run All Quality Gates**

```bash
$ cargo fmt --check
✅ All files formatted

$ cargo clippy -- -D warnings
✅ Zero warnings

$ cargo test
✅ 169 tests passing

$ cargo test --lib
✅ Fast tests: 0.01s
```

**Final REFACTOR Result:**
- Tests: 169 passing ✅
- Clippy: Zero warnings ✅
- Complexity: ≤10 ✅
- Documentation: Complete ✅
- Format: Consistent ✅

## Complete Cycle Example: Random Forest

Let's see a complete RED-GREEN-REFACTOR cycle from aprender's Random Forest implementation.

### RED Phase (7 failing tests)

```rust
#[cfg(test)]
mod random_forest_tests {
    use super::*;

    #[test]
    fn test_random_forest_creation() {
        let rf = RandomForestClassifier::new(10);
        assert_eq!(rf.n_estimators, 10);
    }

    #[test]
    fn test_random_forest_fit() {
        let x = Matrix::from_vec(12, 2, vec![/* iris data */]).unwrap();
        let y = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];

        let mut rf = RandomForestClassifier::new(5);
        assert!(rf.fit(&x, &y).is_ok());
    }

    #[test]
    fn test_random_forest_predict() {
        let x = Matrix::from_vec(12, 2, vec![/* iris data */]).unwrap();
        let y = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];

        let mut rf = RandomForestClassifier::new(5)
            .with_random_state(42);

        rf.fit(&x, &y).unwrap();
        let predictions = rf.predict(&x);

        assert_eq!(predictions.len(), 12);
    }

    #[test]
    fn test_random_forest_reproducible() {
        let x = Matrix::from_vec(12, 2, vec![/* iris data */]).unwrap();
        let y = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];

        let mut rf1 = RandomForestClassifier::new(5).with_random_state(42);
        let mut rf2 = RandomForestClassifier::new(5).with_random_state(42);

        rf1.fit(&x, &y).unwrap();
        rf2.fit(&x, &y).unwrap();

        let pred1 = rf1.predict(&x);
        let pred2 = rf2.predict(&x);

        assert_eq!(pred1, pred2);  // Same seed = same predictions
    }

    #[test]
    fn test_bootstrap_sample_reproducible() {
        let sample1 = _bootstrap_sample(100, Some(42));
        let sample2 = _bootstrap_sample(100, Some(42));
        assert_eq!(sample1, sample2);
    }

    #[test]
    fn test_bootstrap_sample_different_seeds() {
        let sample1 = _bootstrap_sample(100, Some(42));
        let sample2 = _bootstrap_sample(100, Some(123));
        assert_ne!(sample1, sample2);
    }

    #[test]
    fn test_bootstrap_sample_size() {
        let sample = _bootstrap_sample(50, None);
        assert_eq!(sample.len(), 50);
    }
}
```

**Run tests:**
```bash
$ cargo test random_forest
error[E0433]: failed to resolve: could not find `RandomForestClassifier`
# Result: 7/7 tests failed ✅ (expected - not implemented)
```

### GREEN Phase (Minimal Implementation)

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RandomForestClassifier {
    trees: Vec<DecisionTreeClassifier>,
    n_estimators: usize,
    max_depth: Option<usize>,
    random_state: Option<u64>,
}

impl RandomForestClassifier {
    pub fn new(n_estimators: usize) -> Self {
        Self {
            trees: Vec::new(),
            n_estimators,
            max_depth: None,
            random_state: None,
        }
    }

    pub fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
        self
    }

    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    pub fn fit(&mut self, x: &Matrix<f32>, y: &[usize]) -> Result<(), &'static str> {
        self.trees.clear();
        let n_samples = x.shape().0;

        for i in 0..self.n_estimators {
            // Bootstrap sample
            let seed = self.random_state.map(|s| s + i as u64);
            let bootstrap_indices = _bootstrap_sample(n_samples, seed);

            // Extract bootstrap sample
            let (x_boot, y_boot) = extract_bootstrap_samples(x, y, &bootstrap_indices);

            // Train tree
            let mut tree = DecisionTreeClassifier::new();
            if let Some(depth) = self.max_depth {
                tree = tree.with_max_depth(depth);
            }

            tree.fit(&x_boot, &y_boot)?;
            self.trees.push(tree);
        }

        Ok(())
    }

    pub fn predict(&self, x: &Matrix<f32>) -> Vec<usize> {
        let n_samples = x.shape().0;
        let mut predictions = Vec::with_capacity(n_samples);

        for sample_idx in 0..n_samples {
            // Collect votes from all trees
            let mut votes: HashMap<usize, usize> = HashMap::new();

            for tree in &self.trees {
                let tree_prediction = tree.predict(x)[sample_idx];
                *votes.entry(tree_prediction).or_insert(0) += 1;
            }

            // Majority vote
            let prediction = votes
                .into_iter()
                .max_by_key(|&(_, count)| count)
                .map(|(class, _)| class)
                .unwrap_or(0);

            predictions.push(prediction);
        }

        predictions
    }
}

fn _bootstrap_sample(n_samples: usize, random_state: Option<u64>) -> Vec<usize> {
    use rand::distributions::{Distribution, Uniform};
    use rand::SeedableRng;

    let dist = Uniform::from(0..n_samples);
    let mut indices = Vec::with_capacity(n_samples);

    if let Some(seed) = random_state {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        for _ in 0..n_samples {
            indices.push(dist.sample(&mut rng));
        }
    } else {
        let mut rng = rand::thread_rng();
        for _ in 0..n_samples {
            indices.push(dist.sample(&mut rng));
        }
    }

    indices
}
```

**Run tests:**
```bash
$ cargo test random_forest
running 7 tests
test tree::random_forest_tests::test_bootstrap_sample_size ... ok
test tree::random_forest_tests::test_bootstrap_sample_reproducible ... ok
test tree::random_forest_tests::test_bootstrap_sample_different_seeds ... ok
test tree::random_forest_tests::test_random_forest_creation ... ok
test tree::random_forest_tests::test_random_forest_fit ... ok
test tree::random_forest_tests::test_random_forest_predict ... ok
test tree::random_forest_tests::test_random_forest_reproducible ... ok

test result: ok. 7 passed; 0 failed; 0 ignored; 0 measured
# Result: 184 total (177 + 7 new) ✅
```

### REFACTOR Phase

**Step 1: Fix Clippy Warnings**

```bash
$ cargo clippy -- -D warnings
warning: the loop variable `sample_idx` is only used to index `predictions`
  --> src/tree/mod.rs:234:9

# Fix: Add allow annotation (manual indexing is clearer here)
#[allow(clippy::needless_range_loop)]
pub fn predict(&self, x: &Matrix<f32>) -> Vec<usize> {
    // ...
}
```

**Step 2: All Quality Gates**

```bash
$ cargo fmt --check
✅ Formatted

$ cargo clippy -- -D warnings
✅ Zero warnings

$ cargo test
✅ 184 tests passing

$ cargo test --lib
✅ Fast: 0.01s
```

**Final Result:**
- Cycle complete: RED → GREEN → REFACTOR ✅
- Tests: 184 passing (+7) ✅
- TDG: 93.3/100 maintained ✅
- Zero warnings ✅

## Cycle Discipline

**Every feature follows this cycle:**

1. **RED**: Write failing tests
2. **GREEN**: Minimal implementation
3. **REFACTOR**: Comprehensive improvement

**No shortcuts. No exceptions.**

## Benefits of the Cycle

1. **Safety**: Tests catch regressions during refactoring
2. **Clarity**: Tests document expected behavior
3. **Design**: Tests force clean API design
4. **Confidence**: Refactor fearlessly
5. **Quality**: Continuous improvement

## Summary

The RED-GREEN-REFACTOR cycle is:
- **RED**: Write tests FIRST (fail for right reason)
- **GREEN**: Implement MINIMALLY (just pass tests)
- **REFACTOR**: Improve COMPREHENSIVELY (with test safety net)

**Every feature. Every function. Every time.**

**Next:** [Test-First Philosophy](./test-first-philosophy.md)

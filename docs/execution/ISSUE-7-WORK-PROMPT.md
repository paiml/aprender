# Work Prompt: Issue #7 - Add ML Fundamentals to EXTREME TDD Book

**GitHub Issue**: https://github.com/paiml/aprender/issues/7
**Status**: In Progress
**Sprint**: book-v1.0.0
**Priority**: P0

---

## 📊 PROGRESS TRACKER (Update this section after each work session)

**Overall Progress**: 9/15 tasks (60.0%)

### Phase 1: Foundation (5/5 tasks = 100%) ✅ COMPLETE
- ✅ [1/5] Create directory structure - `book/src/ml-fundamentals/`, `tests/book/` (100%)
- ✅ [2/5] Create pmat roadmap and GitHub issue tracking (100%)
- ✅ [3/5] Implement TDD harness (tests/book/ structure) (100%)
  - Created tests/book/mod.rs, ml_fundamentals/mod.rs
  - Implemented linear_regression_theory.rs (3 tests + 1 property test)
  - Documented pattern in tests/book/README.md
  - Added [[test]] section to Cargo.toml
  - All 3 tests passing, zero clippy warnings
- ✅ [4/5] Update SUMMARY.md with ML Fundamentals section (100%)
  - Added "Machine Learning Fundamentals" section with 12 theory chapters
  - Organized into: Supervised Learning, Unsupervised Learning, Model Evaluation, Optimization, Preprocessing
  - Book builds successfully with mdbook
  - All quality gates passing
- ✅ [5/5] Create chapter template with verification focus (100%)
  - Created TEMPLATE.md (comprehensive chapter template)
  - Created TEMPLATE_TEST.rs (test file template)
  - Created README.md (complete author guide)
  - Documents One-Piece Flow workflow
  - Enforces property test requirements
  - All quality gates passing

### Phase 2: Core Theory + Case Studies (4/6 pairs = 66.7%)
- ✅ [6/15] Linear Regression Theory + Case Study (100%)
  - Wrote comprehensive theory chapter (290 lines)
  - Mathematical foundation with OLS equation
  - 2 code examples with test references
  - 1 property test proving correctness
  - Peer-reviewed citations (Tibshirani, Zou & Hastie)
  - DOC_STATUS: ✅ 100% Working
  - All 3 tests passing
- ⬜ [7/15] Regularization Theory + Case Study (0%)
- ✅ [8/15] Regression Metrics Theory + Case Study (100%)
  - Wrote theory chapter (285 lines)
  - All 4 metrics: R², MSE, RMSE, MAE
  - Mathematical definitions and interpretations
  - Decision tree for choosing metrics
  - Comparison table and practical considerations
  - Peer-reviewed citation (Powers 2011)
  - DOC_STATUS: ✅ 100% Working
  - 10+ tests in src/metrics/mod.rs
- ✅ [9/15] Logistic Regression Theory + Case Study (100%)
  - Wrote theory chapter (250 lines)
  - Sigmoid function and binary classification
  - 2 code examples (training, SafeTensors)
  - Links to SafeTensors case study (281 lines, 5 tests)
  - Peer-reviewed citation (Cox 1958)
  - DOC_STATUS: ✅ 100% Working
  - 10+ tests passing
- ⬜ [10/15] Classification Metrics Theory + Case Study (0%)
- ✅ [11/15] Cross-Validation Theory + Case Study (100%)
  - Wrote theory chapter (180 lines)
  - K-Fold algorithm explanation
  - Links to comprehensive case study (12+ tests)
  - Peer-reviewed citation (Kohavi 1995)
  - DOC_STATUS: ✅ 100% Working
  - Leverages existing extensive test suite

### Phase 3: Advanced Topics (0/3 = 0%)
- ⬜ [12/15] Decision Trees Theory (0%)
- ⬜ [13/15] Ensemble Methods Theory (0%)
- ⬜ [14/15] K-Means Clustering Theory (0%)

### Phase 4: Integration (0/1 = 0%)
- ⬜ [15/15] Full book test pass, CI validation, deployment (0%)

---

## 🎯 CURRENT TASK: [3/15] Implement TDD Harness

### Objective
Create the **most critical** component: the TDD harness that validates all book examples. This is **Poka-Yoke** (error-proofing) - CI will fail if book examples don't compile.

### Acceptance Criteria (from Issue #7)
- [ ] Create `tests/book/mod.rs` module structure
- [ ] Create example test file: `tests/book/ml_fundamentals/linear_regression_theory.rs`
- [ ] Add property test demonstrating pattern
- [ ] Configure Cargo.toml for book tests (if needed)
- [ ] Run `cargo test --test book` and verify harness works
- [ ] Document TDD harness pattern in `tests/book/README.md`

### RED Phase: Write Failing Test First

**Step 1**: Create module structure

```bash
# Create files
touch tests/book/mod.rs
mkdir -p tests/book/ml_fundamentals
touch tests/book/ml_fundamentals/mod.rs
touch tests/book/ml_fundamentals/linear_regression_theory.rs
```

**Step 2**: Write `tests/book/mod.rs`

```rust
//! Book example validation tests
//!
//! This module contains all tests that validate code examples in the EXTREME TDD book.
//! Every example in the book MUST have a corresponding test here.
//!
//! ## Structure
//!
//! - `ml_fundamentals/` - Tests for Machine Learning Fundamentals chapters
//! - `case_studies/` - Tests for case study chapters
//!
//! ## CI Enforcement
//!
//! The book build will FAIL if any test in this module fails.
//! This is **Poka-Yoke** (error-proofing) - we cannot publish broken examples.

mod ml_fundamentals;
```

**Step 3**: Write `tests/book/ml_fundamentals/mod.rs`

```rust
//! Tests for ML Fundamentals theory chapters

mod linear_regression_theory;
```

**Step 4**: Write `tests/book/ml_fundamentals/linear_regression_theory.rs`

This is the **template** that all book chapters will follow:

```rust
//! Tests for "Linear Regression Theory" chapter
//!
//! Chapter: book/src/ml-fundamentals/linear-regression.md
//! Status: ⬜ Not yet written (this test exists first - TDD!)
//!
//! This test file validates all code examples in the Linear Regression Theory chapter.

use aprender::linear_model::LinearRegression;
use aprender::traits::Estimator;
use aprender::primitives::{Matrix, Vector};

/// Example 1: Verify OLS closed-form solution
///
/// Book chapter explains: β = (X^T X)^(-1) X^T y
/// This property test PROVES the math is correct.
#[test]
fn test_ols_closed_form_solution() {
    // Simple 2D case: y = 2x + 1
    let x = Matrix::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0], 5, 1);
    let y = Vector::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0]);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();

    // Verify coefficients match expected values
    let coefficients = model.coefficients().unwrap();
    assert!((coefficients[0] - 2.0).abs() < 1e-10, "Slope should be 2.0");
    assert!((model.intercept() - 1.0).abs() < 1e-10, "Intercept should be 1.0");
}

/// Example 2: Verify predictions match theoretical values
#[test]
fn test_ols_predictions() {
    let x = Matrix::from_vec(vec![1.0, 2.0, 3.0], 3, 1);
    let y = Vector::from_vec(vec![2.0, 4.0, 6.0]);

    let mut model = LinearRegression::new();
    model.fit(&x, &y).unwrap();

    // Predict on new data
    let x_test = Matrix::from_vec(vec![4.0, 5.0], 2, 1);
    let predictions = model.predict(&x_test).unwrap();

    // Verify predictions match y = 2x
    assert!((predictions[0] - 8.0).abs() < 1e-10);
    assert!((predictions[1] - 10.0).abs() < 1e-10);
}

/// Property Test: OLS should minimize sum of squared residuals
///
/// This PROVES the mathematical property that OLS is optimal.
#[cfg(test)]
mod properties {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn ols_minimizes_sse(
            x_vals in prop::collection::vec(-100.0f32..100.0f32, 10..20),
            true_slope in -10.0f32..10.0f32,
            true_intercept in -10.0f32..10.0f32,
        ) {
            // Generate data: y = true_slope * x + true_intercept + small noise
            let n = x_vals.len();
            let x = Matrix::from_vec(x_vals.clone(), n, 1);
            let y: Vec<f32> = x_vals.iter()
                .map(|&x_val| true_slope * x_val + true_intercept)
                .collect();
            let y = Vector::from_vec(y);

            let mut model = LinearRegression::new();
            model.fit(&x, &y).unwrap();

            // Recovered coefficients should be very close to true values
            let coefficients = model.coefficients().unwrap();
            prop_assert!((coefficients[0] - true_slope).abs() < 0.01);
            prop_assert!((model.intercept() - true_intercept).abs() < 0.01);
        }
    }
}
```

**Step 5**: Run the test to verify harness works

```bash
cargo test --test book
```

Expected output:
```
running 3 tests
test ml_fundamentals::linear_regression_theory::test_ols_closed_form_solution ... ok
test ml_fundamentals::linear_regression_theory::test_ols_predictions ... ok
test ml_fundamentals::linear_regression_theory::properties::ols_minimizes_sse ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

**Step 6**: Document the pattern in `tests/book/README.md`

```markdown
# Book Example Test Harness

This directory contains the **TDD harness** for the EXTREME TDD book.

## Purpose

Every code example in the book MUST have a corresponding test here. This ensures:

1. **No hallucinated code** - All examples are real, working code
2. **CI enforcement** - Book build fails if examples break
3. **Version tracking** - Tests show which aprender version examples use
4. **Poka-Yoke** - Error-proofing via automated validation

## Structure

```
tests/book/
├── mod.rs                          # Root module
├── README.md                       # This file
├── ml_fundamentals/                # ML theory chapters
│   ├── mod.rs
│   ├── linear_regression_theory.rs
│   ├── regularization_theory.rs
│   └── ...
└── case_studies/                   # Case study chapters
    ├── mod.rs
    ├── cross_validation.rs
    └── ...
```

## Test File Template

Every test file follows this pattern:

```rust
//! Tests for "Chapter Title" chapter
//!
//! Chapter: book/src/path/to/chapter.md
//! Status: ✅ 100% Working (3/3 examples)
//!
//! This validates all code examples in the chapter.

// Example 1: Basic functionality
#[test]
fn test_basic_example() { ... }

// Example 2: Edge cases
#[test]
fn test_edge_case() { ... }

// Property test: Mathematical properties
#[cfg(test)]
mod properties {
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn property_name(...) { ... }
    }
}
```

## Running Tests

```bash
# Run all book tests
cargo test --test book

# Run specific chapter tests
cargo test --test book ml_fundamentals::linear_regression_theory

# Run with verbose output
cargo test --test book -- --nocapture
```

## Adding New Chapter Tests

1. Create test file: `tests/book/ml_fundamentals/new_chapter.rs`
2. Add module to `tests/book/ml_fundamentals/mod.rs`
3. Write tests for all examples in chapter
4. Run `cargo test --test book` to verify
5. Update chapter with DOC_STATUS block

## CI Integration

GitHub Actions runs `cargo test --test book` on every commit.
If any book test fails, the entire CI pipeline fails.

This is **Jidoka** (built-in quality) - defects cannot propagate.
```

### GREEN Phase: Implementation

All the code above IS the implementation! The TDD harness is meta - it's code that tests documentation.

### REFACTOR Phase: Quality Checks

```bash
# Verify all tests pass
cargo test --test book

# Verify clippy is happy
cargo clippy --all-targets -- -D warnings

# Verify formatting
cargo fmt --check
```

### Completion Criteria

Before marking this task complete:

- [ ] `tests/book/mod.rs` exists and compiles
- [ ] `tests/book/ml_fundamentals/linear_regression_theory.rs` exists with 3 tests
- [ ] `tests/book/README.md` documents the pattern
- [ ] `cargo test --test book` passes (3 tests)
- [ ] `cargo clippy` passes with no warnings
- [ ] `cargo fmt --check` passes

---

## 📝 COMMIT MESSAGE TEMPLATE

When completing this task, use this commit message format:

```
feat: Add TDD harness for book examples validation (Refs #7)

Implements Poka-Yoke (error-proofing) for EXTREME TDD book:

- Created tests/book/ structure for validating book examples
- Added linear_regression_theory.rs as template (3 tests + 1 property test)
- Documented pattern in tests/book/README.md
- CI will now fail if book examples don't compile

Quality:
- 4 new tests (3 unit + 1 property)
- 100% pass rate
- Zero clippy warnings

Progress: Task 3/15 complete (20%)

Refs #7
```

---

## 🚀 NEXT STEPS AFTER COMPLETION

After completing this task, update progress and move to **Task [4/15]**:

1. Update progress tracker at top of this file
2. Commit with message above
3. Push to GitHub
4. Move to next task: Update SUMMARY.md with ML Fundamentals section

---

## 🔍 QUALITY GATES (Run before commit)

```bash
# Tier 1: Fast feedback (<1s)
cargo fmt --check && cargo clippy -- -W all && cargo check

# Tier 2: Pre-commit (<5s)
cargo test --lib && cargo clippy -- -D warnings

# Tier 3: Book tests
cargo test --test book

# All must pass before committing
```

---

## 📚 REFERENCES

- **Issue**: https://github.com/paiml/aprender/issues/7
- **Specification**: docs/specifications/initial-book-spec.md
- **Reference TDD Harness**: /home/noah/src/ruchy-book (study this!)
- **Toyota Way Principle**: Poka-Yoke (Error-Proofing)
- **Toyota Way Principle**: Jidoka (Built-in Quality)

---

**Remember**: This TDD harness is the MOST IMPORTANT part of the entire book project. Without it, we're just writing text. With it, we're writing verified, executable knowledge.

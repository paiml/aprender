# Machine Learning Fundamentals - Author Guide

This directory contains ML theory chapters written with **Theory Through Verification** approach.

## 📋 Chapter Template

Use `TEMPLATE.md` as the starting point for all new ML theory chapters.

**Key Principle**: Every mathematical claim must be verified by a property test.

## ✅ Writing a New Chapter - Step-by-Step

### Step 1: Create Test File FIRST (RED Phase)

```bash
# Copy test template
cp tests/book/TEMPLATE_TEST.rs tests/book/ml_fundamentals/new_topic.rs

# Add to tests/book/ml_fundamentals/mod.rs
echo "mod new_topic;" >> tests/book/ml_fundamentals/mod.rs
```

**Edit the test file**:
1. Replace `[Topic]` with actual topic name
2. Replace `Algorithm` with actual struct name
3. Write failing tests for all examples you plan to include
4. Run `cargo test --test book` - tests should FAIL

### Step 2: Create Chapter File (GREEN Phase)

```bash
# Copy chapter template
cp book/src/ml-fundamentals/TEMPLATE.md book/src/ml-fundamentals/new-topic.md
```

**Edit the chapter file**:
1. Replace all `[Topic]` placeholders
2. Write mathematical foundation
3. Add code examples that make the tests PASS
4. Link to test file in each example
5. Update DOC_STATUS block

### Step 3: Verify All Examples Work (GREEN Phase)

```bash
# Run book tests
cargo test --test book ml_fundamentals::new_topic

# All tests should now PASS
```

### Step 4: Add Property Tests (REFACTOR Phase)

1. Write property tests that prove mathematical correctness
2. Add property test references to chapter
3. Verify: `cargo test --test book ml_fundamentals::new_topic::properties`

### Step 5: Update Book Structure

```bash
# Add to SUMMARY.md under appropriate section
# Update DOC_STATUS to ✅ Working

# Build book
cd book && mdbook build

# Verify chapter renders correctly
cd book && mdbook serve
# Open http://localhost:3000
```

## 📊 Doc Status Blocks

Every chapter MUST have a status block at the top:

```markdown
<!-- DOC_STATUS_START -->
**Chapter Status**: ✅ 100% Working (5/5 examples)

| Status | Count | Examples |
|--------|-------|----------|
| ✅ Working | 5 | All examples passing |
| ⏳ In Progress | 0 | - |
| ⬜ Not Implemented | 0 | - |

*Last tested: 2025-11-19*
*Aprender version: 0.3.0*
*Test file: tests/book/ml_fundamentals/new_topic.rs*
<!-- DOC_STATUS_END -->
```

**Status Levels**:
- ✅ **Working**: All examples compile and tests pass
- ⏳ **In Progress**: Some examples work, some don't
- ⬜ **Not Implemented**: Placeholder, no working code yet

## 🔬 Property Test Requirements

Every chapter MUST include at least ONE property test that verifies a mathematical property.

**Good Property Test Example**:
```rust
proptest! {
    #[test]
    fn ols_minimizes_sse(
        x_vals in prop::collection::vec(-100.0f32..100.0f32, 10..20),
        true_slope in -10.0f32..10.0f32,
    ) {
        // Generate perfect linear data
        let n = x_vals.len();
        let x = Matrix::from_vec(n, 1, x_vals.clone()).unwrap();
        let y: Vec<f32> = x_vals.iter()
            .map(|&x_val| true_slope * x_val)
            .collect();

        // OLS should recover true slope exactly
        let mut model = LinearRegression::new();
        model.fit(&x, &y).unwrap();

        let coef = model.coefficients();
        prop_assert!((coef[0] - true_slope).abs() < 0.01);
    }
}
```

**Why This is Good**:
- Tests INFINITE inputs (proptest generates 100+ random cases)
- Verifies mathematical property (OLS recovers true coefficients)
- Proves correctness, not just "works on one example"

## 🚫 Anti-Patterns to Avoid

### ❌ DON'T: Write generic textbook explanations

```markdown
# Bad Example
Linear regression is a supervised learning algorithm that models
the relationship between variables. It was invented in...
[5 paragraphs of history and theory with no code]
```

### ✅ DO: Focus on verification through code

```markdown
# Good Example
Linear regression finds coefficients β that minimize squared error.

**Property Test**: This PROVES OLS is optimal:
[Property test code that verifies the math]

The test shows that for ANY random data, OLS recovers the true
coefficients. This isn't just an example - it's a proof.
```

### ❌ DON'T: Show code examples without tests

```markdown
# Bad Example
Here's how to use linear regression:
[Code block with no test reference]
```

### ✅ DO: Link every example to a test

```markdown
# Good Example
Here's how to use linear regression:
[Code block]

**Test Reference**: `tests/book/ml_fundamentals/linear_regression.rs::test_basic_usage`

If this example breaks, the book build fails. **Poka-Yoke**.
```

## 📚 Required Sections

Every chapter MUST include:

1. ✅ **Mathematical Foundation** - Core equations
2. ✅ **Implementation Examples** - Working code (2-3 examples)
3. ✅ **Property Tests** - At least ONE property test
4. ✅ **Test References** - Link to test file for each example
5. ✅ **DOC_STATUS Block** - Current status
6. ✅ **Practical Considerations** - When to use, performance
7. ✅ **Real-World Application** - Link to case study
8. ✅ **Peer-Reviewed Citation** - At least ONE academic paper

## 🔄 One-Piece Flow (Toyota Way)

Write theory chapter + case study TOGETHER:

1. Start property test (theory)
2. Start case study test
3. Write theory chapter
4. Write case study chapter
5. Both tests pass
6. Deploy together

**Why**: Prevents batch waste. Theory without practice is useless.

## 🎯 Quality Checklist

Before marking a chapter as ✅ Working:

- [ ] All code examples compile
- [ ] All examples have test references
- [ ] At least 1 property test proving math
- [ ] DOC_STATUS block updated
- [ ] `cargo test --test book [chapter]` passes 100%
- [ ] `cargo clippy --all-targets` clean
- [ ] At least 1 peer-reviewed citation
- [ ] Linked to case study
- [ ] Built with `mdbook build` successfully

## 🏆 Example: Linear Regression Theory

See `tests/book/ml_fundamentals/linear_regression_theory.rs` for reference implementation:

- 3 unit tests (basic usage, predictions, edge cases)
- 1 property test (proves OLS minimizes SSE)
- Full documentation
- 100% passing

This is the GOLD STANDARD. All chapters should follow this pattern.

## 📖 Toyota Way Principles Applied

- **Jidoka** (Built-in Quality): Tests prevent defects from propagating
- **Poka-Yoke** (Error-Proofing): CI fails if examples don't compile
- **Kaizen** (Continuous Improvement): Property tests verify math
- **Genchi Genbutsu** (Go and See): Test file shows exact behavior
- **PDCA** (Plan-Do-Check-Act):
  - Plan: Write test (RED)
  - Do: Write chapter (GREEN)
  - Check: Run tests (verify)
  - Act: Refactor and improve

---

**Remember**: Without tests, it's just text. With tests, it's verified knowledge.

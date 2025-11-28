# Mutation Testing

Mutation testing is the most rigorous form of test quality assessment. While code coverage tells you *what* code your tests execute, mutation testing tells you *whether your tests actually verify the code's behavior*.

## The Problem with Coverage Metrics

Consider this code with 100% line coverage:

```rust
pub fn calculate_discount(price: f32, is_member: bool) -> f32 {
    if is_member {
        price * 0.9  // 10% discount
    } else {
        price
    }
}

#[test]
fn test_discount() {
    let result = calculate_discount(100.0, true);
    assert!(result > 0.0);  // Weak assertion!
}
```

This test achieves 100% coverage but would pass even if we changed `0.9` to `0.5` or `1.0`. Mutation testing catches this.

## How Mutation Testing Works

1. **Generate Mutants**: The tool creates variations of your code (mutants)
2. **Run Tests**: Each mutant is tested against your test suite
3. **Kill or Survive**: If tests fail, the mutant is "killed" (good). If tests pass, it "survives" (bad)
4. **Calculate Score**: `Mutation Score = Killed Mutants / Total Mutants`

### Common Mutation Operators

| Operator | Original | Mutant | Tests Should Catch |
|----------|----------|--------|-------------------|
| Arithmetic | `a + b` | `a - b` | Value changes |
| Relational | `a < b` | `a <= b` | Boundary conditions |
| Logical | `a && b` | `a \|\| b` | Boolean logic |
| Literal | `0.9` | `0.0` | Magic numbers |
| Return | `return x` | `return 0` | Return value usage |

## Using cargo-mutants in Aprender

### Installation

```bash
cargo install cargo-mutants --locked
```

### Makefile Targets

Aprender provides tiered mutation testing targets:

```bash
# Quick sample (~5 min) - for rapid feedback
make mutants-fast

# Full suite (~30-60 min) - for comprehensive analysis
make mutants

# Single file - for targeted improvements
make mutants-file FILE=src/metrics/mod.rs

# List potential mutants without running
make mutants-list
```

### Direct Usage

```bash
# Run on entire crate
cargo mutants --no-times --timeout 300 -- --all-features

# Run on specific file
cargo mutants --no-times --timeout 120 --file src/loss/mod.rs

# Run with sharding for CI parallelization
cargo mutants --no-times --shard 1/4 -- --lib
```

## Interpreting Results

### Output Format

```
src/metrics/mod.rs:42: replace mse -> f32 with 0.0 ... KILLED
src/metrics/mod.rs:42: replace mse -> f32 with 1.0 ... KILLED
src/metrics/mod.rs:58: replace mae -> f32 with 0.0 ... SURVIVED  ⚠️
```

### Result Categories

| Status | Meaning | Action |
|--------|---------|--------|
| KILLED | Tests caught the mutation | Good - no action needed |
| SURVIVED | Tests missed the mutation | Add stronger assertions |
| TIMEOUT | Tests took too long | May indicate infinite loop |
| UNVIABLE | Mutant doesn't compile | Normal - skip these |

## Improving Your Mutation Score

### 1. Strengthen Assertions

```rust
// ❌ Weak - survives many mutants
assert!(result > 0.0);

// ✅ Strong - kills most mutants
assert!((result - expected).abs() < 1e-6);
```

### 2. Test Boundary Conditions

```rust
#[test]
fn test_boundaries() {
    // Test exact boundaries, not just general cases
    assert_eq!(classify(0), Category::Zero);
    assert_eq!(classify(1), Category::Positive);
    assert_eq!(classify(-1), Category::Negative);
}
```

### 3. Verify Return Values

```rust
// ❌ Just calling the function
let _ = process_data(&input);

// ✅ Verify the actual result
let result = process_data(&input);
assert_eq!(result.len(), expected_len);
assert!(result.iter().all(|x| *x >= 0.0));
```

### 4. Test Error Paths

```rust
#[test]
fn test_error_handling() {
    // Verify errors are returned, not just that function doesn't panic
    let result = parse_config("invalid");
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("invalid"));
}
```

## Mutation Score Targets

| Project Stage | Target Score | Rationale |
|--------------|--------------|-----------|
| Prototype | 50% | Focus on functionality |
| Development | 70% | Growing confidence |
| Production | 80% | Reliability requirement |
| Critical Path | 90%+ | Zero-defect tolerance |

Aprender targets **85%+ mutation score** for core algorithms.

## CI Integration

### GitHub Actions Example

```yaml
mutation-test:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Install cargo-mutants
      run: cargo install cargo-mutants --locked
    - name: Run mutation tests (sample)
      run: cargo mutants --no-times --shard 1/4 --timeout 300
      continue-on-error: true
    - name: Upload results
      uses: actions/upload-artifact@v4
      with:
        name: mutants-results
        path: mutants.out/
```

### Sharding for Parallelization

```bash
# Split across 4 CI jobs
cargo mutants --shard 1/4  # Job 1
cargo mutants --shard 2/4  # Job 2
cargo mutants --shard 3/4  # Job 3
cargo mutants --shard 4/4  # Job 4
```

## Real Example: Fixing a Surviving Mutant

### The Surviving Mutant

```
src/loss/mod.rs:85: replace - with + in cross_entropy ... SURVIVED
```

### The Original Test

```rust
#[test]
fn test_cross_entropy() {
    let predictions = vec![0.9, 0.1];
    let targets = vec![1.0, 0.0];
    let loss = cross_entropy(&predictions, &targets);
    assert!(loss > 0.0);  // Too weak!
}
```

### The Fix

```rust
#[test]
fn test_cross_entropy_value() {
    let predictions = vec![0.9, 0.1];
    let targets = vec![1.0, 0.0];
    let loss = cross_entropy(&predictions, &targets);

    // Expected: -1.0 * ln(0.9) - 0.0 * ln(0.1) ≈ 0.105
    assert!((loss - 0.105).abs() < 0.01);
}

#[test]
fn test_cross_entropy_increases_with_wrong_prediction() {
    let good_pred = cross_entropy(&[0.9], &[1.0]);
    let bad_pred = cross_entropy(&[0.1], &[1.0]);

    assert!(bad_pred > good_pred);  // Wrong predictions = higher loss
}
```

## Best Practices

1. **Start Small**: Run `mutants-fast` during development
2. **Target High-Risk Code**: Focus on algorithms and business logic
3. **Skip Test Code**: Don't mutate test files themselves
4. **Use Timeouts**: Prevent infinite loops from stalling CI
5. **Review Survivors**: Each surviving mutant is a potential bug

## Relationship to Other Testing

| Test Type | What It Measures | Speed |
|-----------|------------------|-------|
| Unit Tests | Functionality | Fast |
| Property Tests | Invariants | Medium |
| Coverage | Code execution | Fast |
| **Mutation Testing** | Test quality | Slow |

Mutation testing is the final arbiter of test suite quality. Use it to validate that your other testing efforts actually catch bugs.

## See Also

- [What is Mutation Testing?](./what-is-mutation-testing.md)
- [Using cargo-mutants](./using-cargo-mutants.md)
- [Mutation Score Targets](./mutation-score-targets.md)
- [Killing Mutants](./killing-mutants.md)
- [Property-Based Testing](./property-based-testing.md)

# Mutation Testing Methodology

Repartir uses mutation testing as part of the Iron Lotus Framework's Jidoka (automated quality gates) principle.

## What is Mutation Testing?

Mutation testing evaluates the quality of your test suite by:
1. **Mutating** the source code (e.g., changing `>` to `>=`, `+` to `-`)
2. **Running** the test suite against each mutation
3. **Scoring** based on how many mutants are caught by failing tests

## Metrics

**Mutation Score** = (Caught Mutants / Total Mutants) × 100%

- **Caught**: Test suite detected the mutation (tests failed) ✅
- **Survived**: Mutation didn't break tests (weak test coverage) ❌
- **Timeout**: Mutation caused infinite loop (caught) ⏱️
- **Unviable**: Mutation couldn't compile (neutral) ⚠️

## Repartir Targets (Certeza Tier 3)

| Metric                | Target | Iron Lotus Level |
|-----------------------|--------|------------------|
| Mutation Score        | ≥80%   | Required         |
| Mutation Score (goal) | ≥85%   | Excellence       |
| Test Pass Rate        | ≥95%   | Required         |

## Configuration (.cargo-mutants.toml)

```toml
exclude_files = ["tests/**", "benches/**", "examples/**"]
timeout_multiplier = 2.0
minimum_test_pass_rate = 0.95
```

## Running Mutation Tests

### Full Suite (Tier 3)
```bash
# Run all mutations (may take hours)
cargo mutants --no-times --features cpu

# Save results
cargo mutants --no-times --features cpu --output mutation-results.txt
```

### Incremental (Development)
```bash
# Test only modified files since last commit
cargo mutants --in-diff origin/main

# Test specific file
cargo mutants --file src/scheduler/mod.rs
```

### Analysis
```bash
# List all mutations without running
cargo mutants --list

# Show caught mutations
cargo mutants --caught-only

# Show survived mutations (needs investigation)
cargo mutants --unviable
```

## Interpreting Results

### Example Output
```
Mutant 1/100: src/scheduler/mod.rs:44:30: replace == with !=
  CAUGHT in 1.2s

Mutant 2/100: src/task/mod.rs:15:5: replace new() with Default::default()
  SURVIVED in 1.5s

Final score: 85 caught, 10 survived, 5 unviable
Mutation score: 89.5% (85 / 95)
```

### What to Do About Survivors

**High Priority Survivors** (fix immediately):
- Logic operators (`==` → `!=`, `>` → `>=`)
- Arithmetic operators (`+` → `-`, `*` → `/`)
- Boolean conditions (`true` → `false`)

**Medium Priority**:
- Return value replacements (if not covered by integration tests)
- Boundary conditions (off-by-one errors)

**Low Priority** (may be acceptable):
- Constructor alternatives (if semantically equivalent)
- Default implementations (if explicitly tested elsewhere)

## Common Mutation Types

### 1. Binary Operator Replacement
```rust
// Original
if count > threshold { ... }

// Mutations
if count >= threshold { ... }  // Often caught by boundary tests
if count == threshold { ... }  // Should be caught
if count < threshold { ... }   // Should be caught
```

**How to Catch**: Test boundary conditions explicitly
```rust
#[test]
fn test_threshold_boundary() {
    assert!(is_valid(threshold));      // Equal case
    assert!(is_valid(threshold + 1));  // Just above
    assert!(!is_valid(threshold - 1)); // Just below
}
```

### 2. Return Value Replacement
```rust
// Original
fn capacity(&self) -> usize {
    self.num_workers
}

// Mutations
fn capacity(&self) -> usize { 0 }     // Mutant 1
fn capacity(&self) -> usize { 1 }     // Mutant 2
```

**How to Catch**: Assert actual values, not just non-zero
```rust
#[test]
fn test_capacity() {
    let pool = Pool::builder().cpu_workers(4).build();
    assert_eq!(pool.capacity(), 4);  // Catches both mutants
}
```

### 3. Function Body Removal
```rust
// Original
fn shutdown(&mut self) {
    self.workers.clear();
    self.running = false;
}

// Mutation
fn shutdown(&mut self) { }  // Empty function
```

**How to Catch**: Test side effects
```rust
#[test]
fn test_shutdown_clears_workers() {
    let mut pool = create_pool();
    pool.shutdown();
    assert_eq!(pool.worker_count(), 0);
    assert!(!pool.is_running());
}
```

## Repartir Mutation Analysis

### Well-Tested Areas (Expected High Scores)
- **Scheduler**: Priority queue logic (extensive property tests)
- **Task**: Builder pattern validation (comprehensive unit tests)
- **Error handling**: All error paths tested

### Potential Weak Areas (Watch For)
- **Executor capacity**: May have constructor mutants surviving
- **Pool shutdown**: Async cleanup might miss edge cases
- **Remote protocol**: Binary serialization edge cases

## Continuous Improvement

### After Each Mutation Test Run
1. **Analyze survivors**: `cargo mutants --survivors-only`
2. **Add tests**: Write targeted tests for survived mutants
3. **Re-run**: Verify new tests catch the mutants
4. **Commit**: Include mutation score in PR description

### Workflow Integration
```bash
# Pre-commit (quick check)
make tier2

# Pre-merge (full validation)
make tier3  # Includes mutation testing
```

## Comparison with Traditional Coverage

| Metric              | Line Coverage | Mutation Score |
|---------------------|---------------|----------------|
| Measures            | Executed lines| Quality of tests |
| Can be gamed        | Yes (easy)    | No (hard)       |
| False confidence    | High risk     | Low risk        |
| Computation cost    | Low           | High            |
| Industry standard   | ≥80%          | ≥70%            |
| Repartir target     | ≥95%          | ≥80%            |

**Example of Weak Test**:
```rust
// This achieves 100% line coverage but 0% mutation score
#[test]
fn test_add() {
    add(2, 3);  // No assertion!
}
```

## Tools and Resources

- **cargo-mutants**: Mutation testing for Rust
- **Configuration**: `.cargo-mutants.toml`
- **CI Integration**: `.github/workflows/jidoka-gates.yml`

## Academic Foundations

1. **DeMillo et al. (1978)**: "Hints on Test Data Selection"
   - Original mutation testing paper

2. **Jia & Harman (2011)**: "An Analysis and Survey of the Development of Mutation Testing"
   - Comprehensive mutation testing survey

3. **Papadakis et al. (2019)**: "Mutation Testing Advances: An Analysis and Survey"
   - Modern mutation testing techniques

## FAQ

**Q: Why 80% instead of 100%?**
A: Some mutants are equivalent (semantically identical) or redundant. Aiming for 100% leads to diminishing returns.

**Q: How long does mutation testing take?**
A: Approximately (number of mutants × average test time). For repartir (~100 mutants, ~2s tests): ~3-5 minutes.

**Q: Should I run mutation tests locally?**
A: For changed files: yes. For full suite: rely on CI (Tier 3).

**Q: What if a mutant times out?**
A: Good! It means your tests have reasonable timeouts and catch infinite loops.

---

**Last Updated**: v1.1
**Next Review**: After significant test suite changes

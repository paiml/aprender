# Tiered TDD Workflow

Renacer follows a **tiered testing workflow** inspired by the Trueno project, optimizing for rapid TDD cycles while maintaining comprehensive test coverage.

## Philosophy

Not all tests are equal in execution time. Tiered TDD separates tests into three tiers based on speed, allowing developers to run appropriate test suites for different workflows.

**Key Principle:** Run the fastest tests frequently, slower tests less often.

## Three Tiers

### Tier 1: Fast Tests (<5 seconds)

**Purpose:** Immediate feedback during development

**Includes:**
- Unit tests
- Property-based tests (proptest)
- Doctests
- Quick integration tests

**When to Run:**
- After every code change
- Before committing
- During RED-GREEN-REFACTOR cycle

```bash
# Run Tier 1 tests
make test-tier1

# Or manually
cargo test --lib  # Unit tests
cargo test --doc  # Doctests
```

**Example output:**
```text
running 97 tests
test result: ok. 97 passed; 0 failed; 0 ignored; 0 measured
Duration: 2.3s ✅
```

### Tier 2: Medium Tests (<30 seconds)

**Purpose:** Comprehensive integration testing

**Includes:**
- All integration tests
- Full syscall filtering tests
- Multi-process tracing tests
- End-to-end feature tests

**When to Run:**
- Before pushing to remote
- During code review
- After completing a feature

```bash
# Run Tier 2 tests
make test-tier2

# Or manually
cargo test --tests  # All integration tests
```

**Example output:**
```text
running 51 integration tests
test result: ok. 51 passed; 0 failed; 0 ignored
Duration: 24.7s ✅
```

### Tier 3: Slow Tests (<5 minutes)

**Purpose:** Exhaustive quality validation

**Includes:**
- Fuzz testing (short runs)
- Mutation testing
- Long-running stress tests
- Performance benchmarks
- Coverage analysis

**When to Run:**
- Before merging to main
- During release preparation
- Weekly on CI/CD
- Sprint completion

```bash
# Run Tier 3 tests
make test-tier3

# Or manually
cargo fuzz run filter_parser -- -max_total_time=60  # 1 minute fuzz
cargo mutants --in-place -t 180  # 3 minute timeout per mutant
```

**Example output:**
```text
Fuzz tests: 15,347 runs in 60s ✅
Mutation tests: 12/15 mutants caught (80%) ✅
Duration: 4m 32s ✅
```

## Makefile Targets (Sprint 29)

### Configuration

```makefile
# Tiered TDD targets following trueno pattern
.PHONY: test-tier1 test-tier2 test-tier3

# Tier 1: Fast unit tests (<5s)
test-tier1:
	@echo "🔬 Tier 1: Fast tests (<5s)"
	@time cargo test --lib
	@time cargo test --doc

# Tier 2: Integration tests (<30s)
test-tier2:
	@echo "🔧 Tier 2: Integration tests (<30s)"
	@time cargo test --tests

# Tier 3: Fuzz + mutation (<5m)
test-tier3:
	@echo "🚀 Tier 3: Fuzz + mutation (<5m)"
	@time cargo fuzz run filter_parser -- -max_total_time=60 || true
	@time cargo mutants --in-place -t 180 || true
```

## Workflow Examples

### Example 1: Feature Development (RED-GREEN-REFACTOR)

```bash
# RED: Write failing test
vim src/filter.rs  # Add test_new_filter_feature

# Run Tier 1 (fast feedback)
make test-tier1
# ❌ FAIL: test_new_filter_feature

# GREEN: Implement feature
vim src/filter.rs  # Implement feature

# Run Tier 1 again
make test-tier1
# ✅ PASS: All 98 tests

# REFACTOR: Improve code
vim src/filter.rs  # Extract helper function

# Run Tier 1 to ensure no regressions
make test-tier1
# ✅ PASS: All 98 tests

# Before commit: Run Tier 2
make test-tier2
# ✅ PASS: All 149 tests

# Commit
git commit -m "feat: Add new filter feature"
```

### Example 2: Pre-Push Validation

```bash
# Run all tiers before pushing
make test-tier1 && make test-tier2 && make test-tier3

# Or use convenience target
make test-all  # Runs tier1, tier2, tier3 sequentially
```

### Example 3: Sprint Completion

```bash
# Full validation before sprint completion
make test-all
make coverage
pmat analyze complexity
pmat validate-tdg

# If all pass ✅, sprint complete!
```

## Integration with Pre-Commit Hooks

Pre-commit hooks use Tier 1 + selective Tier 2:

```bash
# .git/hooks/pre-commit (simplified)
#!/bin/bash

echo "🔬 Running Tier 1 tests..."
make test-tier1 || exit 1

echo "🔧 Running critical Tier 2 tests..."
cargo test --test core_functionality || exit 1

echo "✅ Tests passed, committing..."
```

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tiered Testing

on: [push, pull_request]

jobs:
  tier1:
    name: Tier 1 (Fast)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run Tier 1 tests
        run: make test-tier1
        timeout-minutes: 1

  tier2:
    name: Tier 2 (Integration)
    runs-on: ubuntu-latest
    needs: tier1
    steps:
      - uses: actions/checkout@v2
      - name: Run Tier 2 tests
        run: make test-tier2
        timeout-minutes: 5

  tier3:
    name: Tier 3 (Exhaustive)
    runs-on: ubuntu-latest
    needs: tier2
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v2
      - name: Run Tier 3 tests
        run: make test-tier3
        timeout-minutes: 10
```

## Performance Targets

| Tier | Target Time | Actual (v0.4.1) | Status |
|------|-------------|-----------------|--------|
| Tier 1 | <5s | 2.3s | ✅ |
| Tier 2 | <30s | 24.7s | ✅ |
| Tier 3 | <5m | 4m 32s | ✅ |

## Test Categorization Guidelines

### Tier 1 Criteria:
- No external dependencies (files, network)
- No subprocess spawning
- No sleep/delays
- Pure computation tests

### Tier 2 Criteria:
- May spawn processes
- May create temporary files
- May use realistic test programs
- Should clean up resources

### Tier 3 Criteria:
- Long-running by nature (fuzz, mutation)
- Performance-sensitive
- Resource-intensive
- Optional on developer machines

## Benefits

1. **Faster Development Cycles**
   - Tier 1 provides instant feedback (2-5s)
   - No waiting for slow tests during TDD

2. **Efficient CI/CD**
   - Fail fast on Tier 1 (saves CI minutes)
   - Only run Tier 3 on main branch

3. **Developer Experience**
   - Clear expectations for test duration
   - No surprise 10-minute test runs

4. **Comprehensive Coverage**
   - All tests still run before merge
   - No quality sacrificed for speed

## Monitoring Test Performance

Track test duration over time:

```bash
# Measure Tier 1 performance
time make test-tier1

# Identify slow tests
cargo test --lib -- --report-time

# If Tier 1 exceeds 5s, investigate:
# - Are integration tests in unit test files?
# - Are there unnecessary sleeps?
# - Can tests be parallelized better?
```

## Future Enhancements

Planned improvements:
- **Tier 0**: Compile-only checks (<1s)
- **Tier 4**: Extended fuzz runs (1 hour+)
- **Smart Test Selection**: Only run affected tiers based on changes
- **Parallel Tier Execution**: Run Tier 1 and Tier 2 concurrently

## Related Patterns

Renacer's tiered TDD is inspired by:
- **Trueno**: Original tiered testing pattern
- **Google Testing Blog**: Test sizes (small/medium/large)
- **Bazel**: Test tags for selective execution
- **pytest**: Markers for test categorization

## Resources

- [Trueno Tiered Testing](https://github.com/paiml/trueno)
- [Google Test Blog: Test Sizes](https://testing.googleblog.com/2010/12/test-sizes.html)
- [Bazel Test Encyclopedia](https://bazel.build/reference/test-encyclopedia)

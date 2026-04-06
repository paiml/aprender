# EXTREME TDD Methodology

Renacer is built using **EXTREME TDD** - a rigorous test-driven development approach that ensures zero defects and complete test coverage.

## The Philosophy

> **"Test EVERYTHING. Trust NOTHING. Verify ALWAYS."**

EXTREME TDD goes beyond standard TDD by requiring:
1. **Tests written first** (RED phase) - NO exceptions
2. **Minimal implementation** (GREEN phase) - Just enough to pass
3. **Comprehensive refactoring** (REFACTOR phase) - With safety net of tests
4. **Property-based testing** - Cover edge cases automatically
5. **Mutation testing** - Verify tests actually catch bugs
6. **Zero tolerance** - All tests pass, zero warnings, always

## The RED-GREEN-REFACTOR Cycle

Every feature in Renacer follows this exact cycle:

### RED Phase: Write Failing Tests

**Rule:** Write integration tests BEFORE any implementation.

Example from Sprint 16 (Regex Filtering):

```rust,ignore
// tests/sprint16_regex_filtering_tests.rs
#[test]
fn test_regex_prefix_pattern() {
    let mut cmd = assert_cmd::cargo::cargo_bin_cmd!("renacer");
    cmd.arg("-e")
        .arg("trace=/^open.*/")
        .arg("--")
        .arg("cat")
        .arg("/dev/null");

    let output = cmd.output().unwrap();
    assert!(output.status.success());

    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("openat("));  // Should match /^open.*/
    assert!(!stdout.contains("close(")); // Should NOT match
}
```

**Verification:** Run tests, confirm they FAIL:
```bash
cargo test --test sprint16_regex_filtering_tests
# Expected: 7/9 tests failed (feature not implemented)
```

### GREEN Phase: Minimal Implementation

**Rule:** Write JUST enough code to make tests pass.

```rust,ignore
// src/filter.rs
pub struct SyscallFilter {
    // ... existing fields ...
    include_regex: Vec<Regex>,  // Add regex support
    exclude_regex: Vec<Regex>,
}

fn parse_regex_pattern(s: &str) -> Option<Result<Regex, regex::Error>> {
    if s.starts_with('/') && s.ends_with('/') && s.len() > 2 {
        let pattern = &s[1..s.len() - 1];
        Some(Regex::new(pattern))
    } else {
        None
    }
}
```

**Verification:** Tests now pass:
```bash
cargo test --test sprint16_regex_filtering_tests
# Result: All 9 tests passing ✅
```

### REFACTOR Phase: Improve & Harden

**Rule:** Add unit tests, property tests, and mutation tests. Fix complexity.

1. **Add Unit Tests** (14 added for regex feature):
```rust,ignore
#[test]
fn test_parse_regex_pattern_valid() {
    assert!(parse_regex_pattern("/^open/").is_some());
    assert!(parse_regex_pattern("/.*at$/").is_some());
}

#[test]
fn test_regex_pattern_case_insensitive() {
    let filter = SyscallFilter::new("trace=/(?i)OPEN/").unwrap();
    assert!(filter.should_trace("openat"));
}
```

2. **Run Clippy** (zero warnings tolerance):
```bash
cargo clippy -- -D warnings
# Fix all warnings, refactor complex code
```

3. **Check Complexity** (≤10 target):
```bash
pmat analyze complexity src/
# All functions ≤10 complexity ✅
```

4. **Run Mutation Tests** (80%+ target):
```bash
cargo mutants
# Verify tests catch injected bugs
```

## Sprint-Based Development

Each feature is a "sprint" with its own test file:

```text
tests/
├── sprint1_mvp_tests.rs          # Basic tracing
├── sprint3_full_syscalls_tests.rs # All 335 syscalls
├── sprint5_dwarf_source_tests.rs  # Source correlation
├── sprint13_function_profiling_tests.rs
├── sprint15_negation_tests.rs
├── sprint16_regex_filtering_tests.rs
└── sprint22_html_output_tests.rs
```

**Each sprint follows RED-GREEN-REFACTOR:**
1. Create `tests/sprintN_feature_tests.rs`
2. Write 5-15 integration tests (RED)
3. Implement feature (GREEN)
4. Add unit tests + refactor (REFACTOR)
5. Verify all quality gates pass
6. Commit with detailed sprint report

## Quality Gates

Before ANY commit, ALL gates must pass:

```bash
# 1. Format check
cargo fmt --check

# 2. Clippy (zero warnings)
cargo clippy -- -D warnings

# 3. All tests pass
cargo test

# 4. Property-based tests
cargo test --test property_based_comprehensive

# 5. Security audit
cargo audit
```

These are enforced via pre-commit hook (`.git/hooks/pre-commit`).

## Real Example: Sprint 16 Complete Cycle

### Initial Commit (RED Phase)
```text
test: Sprint 16 - Add regex filtering tests (RED phase)

Created 9 integration tests for regex pattern matching:
- test_regex_prefix_pattern
- test_regex_suffix_pattern
- test_regex_or_pattern
- test_invalid_regex_error
- test_mixed_regex_and_literal

Result: 7/9 tests failed ✅ (expected - feature not implemented)
```

### Implementation Commit (GREEN Phase)
```text
feat: Sprint 16 - Implement regex filtering (GREEN phase)

Modified src/filter.rs:
- Added include_regex, exclude_regex fields
- Implemented parse_regex_pattern()
- Updated should_trace() for regex matching

Result: All 9 integration tests passing ✅
```

### Final Commit (REFACTOR Phase)
```text
feat: Sprint 16 - Advanced Filtering with Regex Patterns (COMPLETE)

REFACTOR Phase:
- Added 14 unit tests for edge cases
- Fixed clippy warnings (ParseResult type alias)
- Complexity check: all functions ≤10 ✅
- Updated documentation

Final Results:
- Tests: 201 total (178 + 23 new) ✅
- Complexity: ≤10 (max: 8) ✅
- Clippy: Zero warnings ✅
- TDG Score: 94.5/100 maintained ✅
```

## Anti-Hallucination Enforcement

**Book examples MUST be test-backed:**

Every code example in this book is validated by:
1. **Integration tests** - Example commands tested in `tests/sprint*.rs`
2. **GitHub Actions** - CI runs all examples automatically
3. **Test references** - Each example links to validating test

Example:
```bash
# This command is validated by tests/sprint9_filtering_tests.rs
renacer -e trace=file -- cat /etc/hostname
```

**If an example cannot be validated by a test, it MUST NOT be in the book.**

## Property-Based Testing

Beyond unit tests, we use `proptest` for comprehensive edge case coverage:

```rust,ignore
use proptest::prelude::*;

proptest! {
    #[test]
    fn test_filter_never_panics(s in "\\PC*") {
        // Fuzz testing: random input should never panic
        let _ = SyscallFilter::new(&s);
    }
}
```

This generates 670+ test cases automatically, catching edge cases humans miss.

## Mutation Testing

Verify that tests actually catch bugs:

```bash
cargo mutants --in-place
```

Mutants injects bugs (e.g., `>` → `<`, `+` → `-`) and verifies tests fail.

**Target:** 80%+ mutation score (caught mutations / total mutations)

## TDG Score

Toyota Development Grade measures code quality:

```bash
pmat analyze tdg src/
```

**Target:** 94+/100 (A grade)

Metrics:
- Test coverage (weight: 30%)
- Complexity (weight: 25%)
- Documentation (weight: 20%)
- Modularity (weight: 15%)
- Error handling (weight: 10%)

## Summary

EXTREME TDD principles used in Renacer:

1. ✅ **RED-GREEN-REFACTOR** - Every feature, every time
2. ✅ **Sprint-based** - Isolated test files per feature
3. ✅ **Zero tolerance** - All tests pass, zero warnings
4. ✅ **Property testing** - 670+ generated test cases
5. ✅ **Mutation testing** - 80%+ mutation score
6. ✅ **Quality gates** - Pre-commit hook enforces standards
7. ✅ **Anti-hallucination** - All book examples test-backed

This methodology ensures Renacer maintains production-quality with zero defects.

**Next:** [RED-GREEN-REFACTOR Cycle](./red-green-refactor.md) | [Quality Gates](./quality-gates.md)

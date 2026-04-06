# Code Review Process

Code review during migration has unique concerns beyond standard Rust review. Reviewers must verify semantic preservation, check for unsafe code correctness, and validate performance characteristics of transpiled code.

## Review Checklist

### General (All Code)

- [ ] Code compiles with zero warnings (`cargo clippy -- -D warnings`)
- [ ] Tests pass and cover the new code (>= 95%)
- [ ] No unnecessary `unwrap()` or `expect()` in production code
- [ ] Error types are meaningful and actionable
- [ ] Documentation exists for public API

### Migration-Specific

- [ ] Transpiled output matches original behavior (parity tests present)
- [ ] No semantic drift from the source language
- [ ] Dependencies mapped correctly (e.g., numpy operations use trueno)
- [ ] Performance benchmarks show no regression vs original

### Unsafe Code Policy

Unsafe code requires elevated review. Any PR containing `unsafe` must:

1. Document why safe alternatives are insufficient
2. Include a `// SAFETY:` comment explaining the invariants
3. Be reviewed by at least two team members
4. Have dedicated tests exercising the unsafe boundary

```rust
// SAFETY: `data` is guaranteed to be aligned to 32 bytes by the allocator,
// and `len` is bounds-checked by the caller. The pointer is valid for the
// lifetime of the slice.
unsafe {
    std::arch::x86_64::_mm256_load_ps(data.as_ptr())
}
```

## Performance Review

For code on the hot path, verify:

| Check | How to Verify |
|-------|---------------|
| No accidental allocations in loops | Run DHAT or review for `Vec::new()`, `format!()`, `to_string()` |
| SIMD where applicable | Check trueno usage for data-parallel operations |
| Correct backend selection | Verify the 5x PCIe rule for GPU paths |
| Buffer reuse | Look for `clear()` + reuse patterns instead of `new()` |

## Using PMAT in Review

Reviewers can use pmat to quickly assess code quality:

```bash
# Check complexity of changed functions
pmat analyze complexity ./src/changed_module.rs

# Find fault patterns (unwrap, panic, unsafe)
pmat query "changed_function" --faults --include-source
```

## Review Workflow

1. Author runs `make tier2` before submitting (pre-commit checks)
2. CI runs `make tier4` automatically on the PR
3. Reviewer checks pmat analysis and CI results
4. Reviewer verifies parity tests exist for migrated code
5. Two approvals required for unsafe code, one for safe code
6. Merge only after quality gate passes (`batuta stack gate`)

## Common Review Feedback

| Issue | Feedback Template |
|-------|-------------------|
| Missing error context | "Add `.context()` with a descriptive message" |
| Bare unwrap | "Replace with `?` or handle the error explicitly" |
| Missing parity test | "Add a test comparing output to the Python original" |
| Allocation in hot loop | "Consider pre-allocating this buffer outside the loop" |
| Undocumented unsafe | "Add a `// SAFETY:` comment explaining the invariants" |

---

**Navigate:** [Table of Contents](../SUMMARY.md)

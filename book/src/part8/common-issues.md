# Common Issues

This chapter catalogs the most frequently encountered problems when using Batuta for transpilation and migration, organized by category with quick-reference solutions.

## Issue Categories

| Category | Frequency | Typical Severity |
|----------|-----------|-----------------|
| [Transpilation Failures](./transpilation-failures.md) | High | Blocking |
| [Type Inference Problems](./type-inference.md) | High | Moderate |
| [Lifetime Errors](./lifetime-errors.md) | Medium | Moderate |
| [Performance Regressions](./performance-regressions.md) | Low | High impact |

## Quick Diagnostic Commands

When something goes wrong, start with these commands to gather context:

```bash
# Check pipeline status and last error
batuta status

# Inspect the current workflow state
batuta report

# Verify tool availability
batuta analyze --check-tools

# Check stack health
batuta stack check
```

## Top 5 Issues and Quick Fixes

### 1. "Tool not found: depyler"

The transpiler binary is not on PATH.

```bash
cargo install depyler
# Or check PATH includes ~/.cargo/bin
echo $PATH | tr ':' '\n' | grep cargo
```

### 2. "Type mismatch in transpiled output"

Dynamic Python types mapped to wrong Rust types. See [Type Inference Problems](./type-inference.md).

```bash
# Re-run with explicit type annotations
batuta transpile --type-hints ./src
```

### 3. "Borrow checker error in C migration"

Ownership model mismatch from C pointers. See [Lifetime Errors](./lifetime-errors.md).

### 4. "Transpiled code slower than original"

Usually caused by missing SIMD engagement or excessive allocation. See [Performance Regressions](./performance-regressions.md).

```bash
# Quick check: is SIMD enabled?
rustc --print cfg | grep target_feature
```

### 5. "Pipeline stuck in validation phase"

The previous phase wrote invalid state. Reset and re-run:

```bash
batuta reset --phase validation
batuta validate --trace
```

## Environment Checklist

Before reporting an issue, verify your environment:

| Requirement | Check Command | Expected |
|-------------|---------------|----------|
| Rust toolchain | `rustc --version` | 1.75+ |
| Cargo | `cargo --version` | Matches rustc |
| LLVM tools | `llvm-cov --version` | 14+ |
| Target CPU features | `rustc --print cfg` | `avx2` or `neon` |
| Transpiler tools | `which depyler decy bashrs` | Paths printed |

See [Debugging Techniques](./debugging.md) and [Getting Help](./getting-help.md) for further assistance.

---

**Navigate:** [Table of Contents](../SUMMARY.md)

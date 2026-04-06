# Phase 4: Validation

Phase 4 verifies that transpiled code preserves the semantic behavior of the original source through multiple independent validation methods.

## Overview

Validation is the critical quality gate before deployment. It answers: "Does the transpiled code do the same thing as the original?"

```
Original Binary ──┬── Syscall Trace ──┐
                  ├── Stdout Capture ──┤── Compare ── Pass/Fail
Transpiled Binary ┬── Syscall Trace ──┘             │
                  ├── Stdout Capture ──────────────┘
                  ├── cargo test ───── Test Results ──┘
                  └── Timing ──── Benchmark Report ───┘
```

## Validation Methods

### 1. Syscall Tracing (Renacer)

The deepest validation: traces system calls made by both binaries using the Renacer tracer. If the syscall sequences match, the programs exhibit equivalent OS-level behavior.

```bash
batuta validate --trace-syscalls
```

Uses `ValidationStage` from the pipeline library, which creates a Tokio runtime to execute the async tracing comparison.

### 2. Output Comparison

Runs both binaries and compares stdout line-by-line. Differences are displayed in a unified diff format (truncated to 20 lines). This catches functional regressions where the program logic diverges.

```bash
batuta validate --diff-output
```

### 3. Test Suite Execution

Runs `cargo test` in the transpiled output directory. This validates that any tests generated during transpilation (or manually added) pass. The output directory is read from `batuta.toml` (`transpilation.output_dir`).

```bash
batuta validate --run-original-tests
```

### 4. Performance Benchmarking

Times both binaries over 3 iterations and reports the average execution time and speedup factor. This is informational — performance regression does not fail the validation phase.

```bash
batuta validate --benchmark
```

## Jidoka Stop-on-Error

Each validation method independently contributes to the overall pass/fail result. If any enabled method detects a mismatch:

1. The Validation phase is marked as **failed** in the workflow state
2. The failure reason is recorded
3. Phase 5 (Build) will refuse to start until validation passes

Missing binaries (for syscall tracing, diff, or benchmark) are treated as **warnings**, not failures. This allows validation to proceed even in environments where the original binary is not available.

## CLI Reference

See [`batuta validate`](../part6/cli-validate.md) for full command documentation.

---

**Previous:** [Phase 3: Optimization](./phase3-optimization.md)
**Next:** [Phase 5: Deployment](./phase5-deployment.md)

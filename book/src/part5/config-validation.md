# Validation Configuration

The `[validation]` section controls Phase 4: semantic equivalence checking between the original program and the transpiled Rust output.

## Top-Level Settings

```toml
[validation]
trace_syscalls = true
run_original_tests = true
diff_output = true
benchmark = false
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `trace_syscalls` | bool | `true` | Record and compare syscall traces via Renacer |
| `run_original_tests` | bool | `true` | Execute the original project's test suite against transpiled code |
| `diff_output` | bool | `true` | Generate unified diff of stdout/stderr between original and transpiled runs |
| `benchmark` | bool | `false` | Run performance benchmarks after validation |

## Syscall Trace Comparison

When `trace_syscalls` is enabled, Batuta invokes Renacer to capture the syscall sequences of both the original and transpiled programs. The traces are compared structurally: matching syscall names, argument patterns, and return values. Divergences are reported as validation warnings.

This is the strongest form of behavioral equivalence checking available in the pipeline.

## Renacer Configuration

```toml
[validation.renacer]
trace_syscalls = []
output_format = "json"
```

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `trace_syscalls` | array | `[]` | Specific syscalls to trace (empty means all) |
| `output_format` | string | `"json"` | Trace output format: `"json"` or `"text"` |

### Filtering Syscalls

When tracing all syscalls produces too much noise, restrict the set to the calls that matter for your application.

```toml
[validation.renacer]
trace_syscalls = ["read", "write", "open", "close", "mmap"]
output_format = "json"
```

## Numerical Tolerance

Floating-point results may differ between the original runtime and the transpiled Rust code due to instruction ordering, fused multiply-add availability, or different math library implementations. Batuta applies a default relative tolerance of 1e-6 when comparing numeric outputs in diff mode.

To adjust tolerance for specific comparisons, use the `--tolerance` flag on the CLI:

```bash
batuta validate --tolerance 1e-4
```

## Benchmark Settings

When `benchmark = true`, Batuta runs the transpiled binary through a timing harness after validation passes. Results are stored in `.batuta-state.json` and included in the report.

```bash
# Enable benchmarks for a single run without changing the config file
BATUTA_VALIDATION_BENCHMARK=true batuta validate
```

---

**Navigate:** [Table of Contents](../SUMMARY.md)

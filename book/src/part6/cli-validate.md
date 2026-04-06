# `batuta validate`

Validate semantic equivalence between original and transpiled code (Phase 4).

## Synopsis

```bash
batuta validate [OPTIONS]
```

## Description

The validate command verifies that transpiled Rust code produces equivalent behavior to the original source. It supports four validation methods: syscall tracing via Renacer, output diffing, test suite execution, and performance benchmarking.

This is Phase 4 of the 5-phase transpilation pipeline. It requires Phase 3 (Optimization) to be completed first.

## Options

| Option | Description |
|--------|-------------|
| `--trace-syscalls` | Trace syscalls for comparison using Renacer |
| `--diff-output` | Compare stdout of original vs transpiled binary |
| `--run-original-tests` | Run `cargo test` in the transpiled output directory |
| `--benchmark` | Run performance benchmarks (3 iterations, reports speedup) |
| `-v, --verbose` | Enable verbose output |
| `-h, --help` | Print help |

## Validation Methods

### Syscall Tracing (`--trace-syscalls`)

Uses the Renacer syscall tracer to compare system call patterns between the original and transpiled binaries. This provides the deepest semantic equivalence guarantee.

**Requires:** `./original_binary` and `./target/release/transpiled` to exist.

### Output Diff (`--diff-output`)

Runs both binaries and compares their stdout line-by-line. Shows a unified diff if outputs differ.

### Test Execution (`--run-original-tests`)

Runs `cargo test` in the transpiled output directory (from `batuta.toml` `transpilation.output_dir`). Validates that the transpiled code passes its test suite.

### Benchmarking (`--benchmark`)

Times both original and transpiled binaries over 3 iterations and reports average execution time and speedup factor.

## Examples

### Full Validation Suite

```bash
$ batuta validate --trace-syscalls --diff-output --run-original-tests --benchmark

✅ Validating equivalence...

Validation Settings:
  • Syscall tracing: enabled
  • Diff output: enabled
  • Original tests: enabled
  • Benchmarks: enabled

🔍 Running Renacer syscall tracing...
  ✅ Syscall traces match - semantic equivalence verified

📊 Output comparison:
  ✅ Outputs match - functional equivalence verified

🧪 Running test suite on transpiled code:
  ✅ All tests pass on transpiled code

⚡ Performance benchmarking:
  Original:   142.3ms avg
  Transpiled:  28.1ms avg
  Speedup:    5.06x faster
```

### Quick Test-Only Validation

```bash
$ batuta validate --run-original-tests
```

### Benchmark Comparison

```bash
$ batuta validate --benchmark
```

## Exit Behavior

Each validation method independently updates the overall pass/fail status. If any enabled method fails, the Validation phase is marked as failed in the workflow state.

If binaries are not found for `--trace-syscalls`, `--diff-output`, or `--benchmark`, those checks are skipped with a warning (not treated as failures).

## See Also

- [Phase 4: Validation](../part2/phase4-validation.md)
- [Syscall Tracing](../part2/syscall-tracing.md)
- [`batuta build`](./cli-build.md) - Next phase

---

**Previous:** [`batuta optimize`](./cli-optimize.md)
**Next:** [`batuta build`](./cli-build.md)
